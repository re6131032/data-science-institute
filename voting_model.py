import numpy as np
import torch
import xgboost as xgb
import pandas as pd
from transformers import BertForSequenceClassification
from sklearn.metrics import classification_report
import json
import pickle
import os
from datetime import datetime
from tqdm import tqdm

# 設置GPU記憶體管理
torch.backends.cuda.max_split_size_mb = 512
if torch.cuda.is_available():
    torch.cuda.empty_cache()

class BertPredictor:
    """BERT模型預測器"""
    def __init__(self, model, device, batch_size=16):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model.eval()
        print("BERT predictor initialized")
        
    def predict_proba(self, input_ids, attention_masks):
        all_probs = []
        n_samples = len(input_ids)
        
        for i in tqdm(range(0, n_samples, self.batch_size), desc="BERT prediction"):
            batch_end = min(i + self.batch_size, n_samples)
            b_input_ids = input_ids[i:batch_end].to(self.device)
            b_mask = attention_masks[i:batch_end].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.vstack(all_probs)

class XGBoostPredictor:
    """XGBoost模型預測器"""
    def __init__(self, model, model_type, target_mapping=None):
        self.model = model
        self.model_type = model_type  # "4cls" or "22cls"
        self.target_mapping = target_mapping
        self.feature_names = [f.strip() for f in self.model.feature_names]
        print(f"XGBoost {model_type} predictor initialized")
    
    def predict_proba(self, features):
        features = features.reindex(columns=self.feature_names)
        dmatrix = xgb.DMatrix(features, feature_names=self.feature_names)
        raw_pred = self.model.predict(dmatrix)
        
        # 將預測轉換到26類空間
        n_samples = len(features)
        probas = np.zeros((n_samples, 26))
        
        # 根據輸出形狀調整處理方式
        if len(raw_pred.shape) == 1:  # 如果是單一預測值
            if self.model_type == "4cls":
                # 4類模型的映射（固定映射）
                target_indices = [5, 19, 0, 17]  # socialanxiety, mentalhealth, COVID19_support, lonely
                temp_probs = np.zeros((n_samples, 4))
                for i in range(len(raw_pred)):
                    temp_probs[i, int(raw_pred[i])] = 1
                raw_pred = temp_probs
                for i in range(n_samples):
                    for j, target_idx in enumerate(target_indices):
                        probas[i, target_idx] = raw_pred[i, j]
            else:
                # 22類模型的映射
                for i in range(len(raw_pred)):
                    mapped_idx = int(self.target_mapping['new_to_original'][str(int(raw_pred[i]))])
                    probas[i, mapped_idx] = 1.0
        else:  # 如果是概率矩陣
            if self.model_type == "4cls":
                # 4類模型的映射
                target_indices = [5, 19, 0, 17]  # socialanxiety, mentalhealth, COVID19_support, lonely
                for i in range(n_samples):
                    for j, target_idx in enumerate(target_indices):
                        probas[i, target_idx] = raw_pred[i, j]
            else:
                # 22類模型的映射
                for i in range(n_samples):
                    for j in range(raw_pred.shape[1]):
                        mapped_idx = int(self.target_mapping['new_to_original'][str(j)])
                        probas[i, mapped_idx] = raw_pred[i, j]
        
        return probas
        

class ConfidenceBasedCombiner:
    """基於信心值的模型組合器"""
    def __init__(self, bert_model, xgb_4cls_model, xgb_22cls_model, device, label_encoder):
        self.label_encoder = label_encoder
        print("\nInitializing model combiners...")
        
        # 初始化預測器
        self.bert_predictor = BertPredictor(bert_model, device)
        
        # 初始化4類XGBoost
        self.xgb_4cls_predictor = XGBoostPredictor(xgb_4cls_model, "4cls")
        
        # 載入22類XGBoost的映射
        with open('base_models/xgboost_22cls/config.json', 'r') as f:
            xgb_22cls_config = json.load(f)
        self.xgb_22cls_predictor = XGBoostPredictor(xgb_22cls_model, "22cls", 
                                                   xgb_22cls_config['target_mapping'])
    
    def combine_predictions(self, bert_proba, xgb_4cls_proba, xgb_22cls_proba):
        """根據BERT預測類別和信心值進行複合決策"""
        print("\nCombining predictions with updated decision logic...")
        
        n_samples = len(bert_proba)
        final_proba = np.zeros_like(bert_proba)
        final_pred = []
        
        # 定義特殊處理的類別
        special_categories = {5, 19, 17}  # socialanxiety, mentalhealth, lonely
        confidence_threshold = 0.6
        
        # 計數器
        stats = {
            'special_low_conf': 0,    # 特殊類別低信心
            'special_high_conf': 0,   # 特殊類別高信心
            'normal_low_conf': 0,     # 一般類別低信心
            'normal_high_conf': 0     # 一般類別高信心
        }
        
        for i in range(n_samples):
            # 獲取BERT的預測和信心值
            bert_pred = np.argmax(bert_proba[i])
            bert_conf = bert_proba[i][bert_pred]
            
            if bert_pred in special_categories:
                # 特殊類別的處理
                if bert_conf < confidence_threshold:
                    # 低信心時結合BERT和XGBoost_22cls的預測
                    stats['special_low_conf'] += 1
                    # 各給0.5的權重
                    final_proba[i] = 0.7 * bert_proba[i] + 0.3 * xgb_22cls_proba[i]
                    final_pred.append(np.argmax(final_proba[i]))
                else:
                    # 高信心時完全採用BERT的預測
                    stats['special_high_conf'] += 1
                    final_proba[i] = bert_proba[i]
                    final_pred.append(bert_pred)
            else:
                # 非特殊類別的處理（保持原有邏輯）
                if bert_conf < confidence_threshold:
                    # 低信心時結合BERT和XGBoost_4cls的預測
                    stats['normal_low_conf'] += 1
                    final_proba[i] = 0.7 * bert_proba[i] + 0.3* xgb_4cls_proba[i]
                    final_pred.append(np.argmax(final_proba[i]))
                else:
                    # 高信心時完全採用BERT的預測
                    stats['normal_high_conf'] += 1
                    final_proba[i] = bert_proba[i]
                    final_pred.append(bert_pred)
        
        # 輸出統計資訊
        print("\nDecision Statistics:")
        print(f"Special Categories (socialanxiety, mentalhealth, lonely):")
        print(f"- Low confidence (BERT:0.5, XGB22:0.5): {stats['special_low_conf']}")
        print(f"- High confidence (trusted BERT): {stats['special_high_conf']}")
        print(f"\nNormal Categories:")
        print(f"- Low confidence (BERT:0.3, XGB4:0.7): {stats['normal_low_conf']}")
        print(f"- High confidence (trusted BERT): {stats['normal_high_conf']}")
        
        return np.array(final_pred), final_proba
    
    def predict(self, input_ids, attention_masks, features):
        """進行預測"""
        print("\nGetting BERT predictions...")
        bert_proba = self.bert_predictor.predict_proba(input_ids, attention_masks)
        
        print("\nGetting 4-class XGBoost predictions...")
        xgb_4cls_proba = self.xgb_4cls_predictor.predict_proba(features)
        
        print("\nGetting 22-class XGBoost predictions...")
        xgb_22cls_proba = self.xgb_22cls_predictor.predict_proba(features)
        
        print("\nCombining predictions...")
        final_pred, final_proba = self.combine_predictions(bert_proba, xgb_4cls_proba, xgb_22cls_proba)
        
        # 計算最終信心值
        final_confidences = np.max(final_proba, axis=1)
        
        return final_pred, final_proba, final_confidences

def load_models_and_data():
    """載入模型和資料"""
    print("Loading models and data...")
    
    # 載入BERT模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    bert_model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=26
    ).to(device)
    bert_model.load_state_dict(torch.load('base_models/bert/model_state.bin'))
    bert_model.eval()
    print("BERT model loaded")
    
    # 載入4類XGBoost模型
    xgb_4cls_model = xgb.Booster()
    xgb_4cls_model.load_model('base_models/targeted_xgboost/model.json')
    print("4-class XGBoost model loaded")
    
    # 載入22類XGBoost模型
    xgb_22cls_model = xgb.Booster()
    xgb_22cls_model.load_model('base_models/xgboost_22cls/model.json')
    print("22-class XGBoost model loaded")
    
    # 載入資料
    print("Loading data...")
    tokenized_data = torch.load('data/processed/tokenized_data.pt')
    features_data = pd.read_csv('data/processed/features_data.csv')
    split_indices = torch.load('data/processed/split_indices.pt')
    
    with open('base_models/encoder/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("Data loaded")
        
    return (bert_model, xgb_4cls_model, xgb_22cls_model, device, tokenized_data, 
            features_data, split_indices, label_encoder)

def save_results(predictions, confidences, metrics, label_encoder, save_dir='final_model'):
    """保存結果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 轉換numpy類型
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {key: convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_native_types(obj.tolist())
        else:
            return obj
    
    results = {
        'metrics': convert_to_native_types(metrics),
        'confidence_analysis': {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        },
        'class_confidence': {
            label_encoder.classes_[i]: float(np.mean(confidences[predictions == i]))
            for i in range(len(label_encoder.classes_))
        },
        'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {save_dir}")

def main():
    try:
        print("=== Starting Three-Model Combination Process ===")
        print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 載入模型和資料
        (bert_model, xgb_4cls_model, xgb_22cls_model, device, tokenized_data, 
         features_data, split_indices, label_encoder) = load_models_and_data()
        
        # 準備測試資料
        print("\nPreparing test data...")
        test_indices = split_indices['test_indices']
        X_test_features = features_data.drop('subreddit', axis=1).iloc[test_indices]
        X_test_ids = tokenized_data['input_ids'][test_indices]
        X_test_masks = tokenized_data['attention_masks'][test_indices]
        y_test = tokenized_data['labels'][test_indices]
        
        print(f"Test data shapes:")
        print(f"Features: {X_test_features.shape}")
        print(f"Input IDs: {X_test_ids.shape}")
        print(f"Labels: {y_test.shape}")
        
        # 初始化組合器
        print("\nInitializing combiner...")
        combiner = ConfidenceBasedCombiner(
            bert_model, xgb_4cls_model, xgb_22cls_model, device, label_encoder
        )
        
        # 進行預測
        print("\nMaking predictions...")
        predictions, probas, confidences = combiner.predict(
            X_test_ids, X_test_masks, X_test_features
        )
        
        # 計算指標
        print("\nCalculating metrics...")
        report = classification_report(
            y_test, 
            predictions,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        # 打印結果
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                 target_names=label_encoder.classes_))
        
        # 保存結果
        save_results(predictions, confidences, report, label_encoder)
        
        print(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()