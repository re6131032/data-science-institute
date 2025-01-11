import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import json
import pickle
import os
from datetime import datetime
import time
import gc
import copy
import psutil
import numpy as np

def format_time(elapsed):
    """把秒数转换为 hh:mm:ss 格式"""
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def print_memory_usage():
    """打印當前記憶體使用狀況"""
    process = psutil.Process(os.getpid())
    print(f"\nCurrent memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    # if torch.cuda.is_available():
    #     print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    #     print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

def clean_memory():
    """清理記憶體"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory_usage()

class BertWithDropout(BertForSequenceClassification):
    """自定義BERT模型，添加dropout層"""
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(0.3)  # 設置dropout rate為0.3
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return type('Output', (), {'loss': loss, 'logits': logits})()
        return type('Output', (), {'logits': logits})()

def load_data():
    """加载预处理好的数据"""
    print("Loading tokenized data...")
    training_data = torch.load('data/processed/tokenized_data.pt')
    
    # 获取数据
    input_ids = training_data['input_ids']
    attention_masks = training_data['attention_masks']
    labels = training_data['labels']
    label_encoder = training_data['label_encoder']
    
    # 加载训练/测试集索引
    split_indices = torch.load('data/processed/split_indices.pt')
    train_indices = split_indices['train_indices']
    test_indices = split_indices['test_indices']
    
    clean_memory()  # 清理載入數據後的記憶體
    
    return input_ids, attention_masks, labels, label_encoder, train_indices, test_indices

def prepare_dataloaders(input_ids, attention_masks, labels, train_indices, test_indices, batch_size=32):
    """准备DataLoader"""
    # 使用 pin_memory 加速数据传输
    train_data = torch.utils.data.TensorDataset(
        input_ids[train_indices],
        attention_masks[train_indices],
        labels[train_indices]
    )
    
    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4  # 使用多進程加載數據
    )
    
    test_data = torch.utils.data.TensorDataset(
        input_ids[test_indices],
        attention_masks[test_indices],
        labels[test_indices]
    )
    
    test_dataloader = DataLoader(
        test_data,
        sampler=SequentialSampler(test_data),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )
    
    clean_memory()  # 清理準備數據後的記憶體
    
    return train_dataloader, test_dataloader

def train_epoch(model, train_dataloader, device, optimizer, scheduler):
    """訓練一個epoch並輸出詳細的loss資訊"""
    total_loss = 0
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        # 進度報告與loss輸出
        if step % 50 == 0:
            avg_loss = total_loss / (step + 1) if step > 0 else 0
            print(f'Batch {step:>5,} of {len(train_dataloader):>5,}. Average loss: {avg_loss:.4f}')
            print_memory_usage()
        
        # 準備數據
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # 清除梯度
        model.zero_grad()
        
        # 前向傳播
        outputs = model(
            b_input_ids,
            attention_mask=b_input_mask,
            labels=b_labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # 反向傳播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 更新參數
        optimizer.step()
        scheduler.step()
        
        # 清理不需要的張量
        del b_input_ids, b_input_mask, b_labels, outputs, loss
        if step % 100 == 0:
            clean_memory()
    
    avg_train_loss = total_loss / len(train_dataloader)
    return avg_train_loss

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            outputs = model(
                b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            
            total_loss += outputs.loss.item()
            
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            labels = b_labels.cpu().numpy()
            
            predictions.extend(pred)
            true_labels.extend(labels)
            
            # 清理不需要的張量
            del b_input_ids, b_input_mask, b_labels, outputs, logits, pred, labels
            clean_memory()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    return accuracy, avg_loss, report, predictions, true_labels

def train_with_early_stopping(model, train_dataloader, test_dataloader, device, optimizer, scheduler,
                            epochs=5, patience=2, save_path='base_models/bert/best_model.bin'):
    """使用early stopping的训练循环"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    training_stats = []
    
    print("Starting training...")
    for epoch_i in range(epochs):
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
        t0 = time.time()
        
        # 训练
        avg_train_loss = train_epoch(model, train_dataloader, device, optimizer, scheduler)
        training_time = format_time(time.time() - t0)
        
        print(f"Average training loss: {avg_train_loss}")
        
        # 验证
        print("Running evaluation...")
        t0 = time.time()
        accuracy, avg_val_loss, report, predictions, true_labels = evaluate(
            model, test_dataloader, device
        )
        validation_time = format_time(time.time() - t0)
        
        print(f"Validation loss: {avg_val_loss}")
        print(f"Accuracy: {accuracy}")
        print("\nClassification Report:")
        print(report)
        
        # Early Stopping 检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = copy.deepcopy(model)
            # 保存最佳模型
            torch.save(model.state_dict(), save_path)
            print(f"Found better model! Saving to {save_path}")
        else:
            patience_counter += 1
            print(f"Validation loss didn't improve. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
        
        # 記錄統計數據
        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Accuracy': accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })
        
        clean_memory()  # 清理每個epoch後的記憶體
    
    return training_stats, best_model

def save_model(model, training_args, metrics, save_dir='base_models/bert'):
    """保存模型和相关信息"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型状态
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_state_1.bin'))
    
    # 保存配置信息
    config = {
        'model_name': 'bert-base-uncased',
        'training_args': training_args,
        'metrics': metrics,
        'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_version': '1.0'
    }
    
    with open(os.path.join(save_dir, 'config_1.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Model saved to {save_dir}")
    clean_memory()

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} for training')
    
    try:
        # 加载数据
        input_ids, attention_masks, labels, label_encoder, train_indices, test_indices = load_data()
        
        # 准备dataloaders
        train_dataloader, test_dataloader = prepare_dataloaders(
            input_ids, attention_masks, labels, 
            train_indices, test_indices
        )
        
        # 初始化自定义模型
        model = BertWithDropout.from_pretrained(
            'bert-base-uncased',
            num_labels=len(label_encoder.classes_)
        )
        model = model.to(device)
        
        # 设置训练参数
        epochs = 5  # 設置為5個epoch
        total_steps = len(train_dataloader) * epochs
        
        optimizer = AdamW(model.parameters(),
                         lr=2e-5,
                         eps=1e-8)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 使用early stopping進行訓練
        training_stats, best_model = train_with_early_stopping(
            model,
            train_dataloader,
            test_dataloader,
            device,
            optimizer,
            scheduler,
            epochs=5,
            patience=2
        )
        
        # 使用最佳模型進行最終評估
        print("\nRunning final evaluation...")
        accuracy, avg_val_loss, report, predictions, true_labels = evaluate(
            best_model, test_dataloader, device
        )
        
        # 保存模型
        training_args = {
            'epochs': epochs,
            'batch_size': 32,
            'learning_rate': 2e-5,
            'max_length': 512,
            'dropout_rate': 0.3
        }
        
        metrics = {
            'final_accuracy': accuracy,
            'final_loss': avg_val_loss,
            'early_stopping_epoch': len(training_stats)
        }
        
        save_model(best_model, training_args, metrics)
        print("Training completed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    finally:
        clean_memory()  # 最終清理記憶體

if __name__ == "__main__":
    main()