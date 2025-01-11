import pandas as pd
import numpy as np
import torch
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import classification_report
import xgboost as xgb

def load_data():
    """載入數據和索引"""
    print("\nLoading data...")
    print("Reading features data...")
    features_data = pd.read_csv('data/processed/features_data.csv')
    print(f"Features data shape: {features_data.shape}")
    
    print("Loading split indices...")
    split_indices = torch.load('data/processed/split_indices.pt')
    train_indices = split_indices['train_indices']
    test_indices = split_indices['test_indices']

    print("Loading label encoder...")
    with open('base_models/encoder/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return features_data, train_indices, test_indices, label_encoder

def prepare_targeted_data(features_data, train_indices, test_indices, label_encoder):
    """準備針對22個類別的訓練數據"""
    print("\nPreparing targeted training data...")
    # 定義要排除的類別
    excluded_categories = ['socialanxiety', 'mentalhealth', 'COVID19_support', 'lonely']
    
    # 獲取其他22個類別
    all_categories = label_encoder.classes_.tolist()
    target_categories = [cat for cat in all_categories if cat not in excluded_categories]
    
    # 建立目標類別到新索引的映射
    target_to_index = {cat: idx for idx, cat in enumerate(target_categories)}
    
    # 取得訓練集的資料
    train_data = features_data.iloc[train_indices]
    
    # 只選擇目標類別的資料
    train_mask = train_data['subreddit'].isin(target_categories)
    train_data = train_data[train_mask]
    
    # 準備特徵和標籤
    X_train = train_data.drop('subreddit', axis=1)
    y_train = train_data['subreddit'].map(target_to_index)
    
    # 測試集保持完整
    test_data = features_data.iloc[test_indices]
    X_test = test_data.drop('subreddit', axis=1)
    y_test = label_encoder.transform(test_data['subreddit'])
    
    # 保存映射關係，供之後使用，確保使用Python原生類型
    global TARGET_MAPPING
    TARGET_MAPPING = {
        'original_to_new': {
            int(label_encoder.transform([cat])[0]): target_to_index[cat] 
            for cat in target_categories
        },
        'new_to_original': {
            target_to_index[cat]: int(label_encoder.transform([cat])[0])
            for cat in target_categories
        }
    }
    
    print('\nTarget categories distribution:')
    values, counts = np.unique(train_data['subreddit'], return_counts=True)
    for v, c in zip(values, counts):
        print(f'{v}: {c} (new index: {target_to_index[v]})')
    
    print(f"\nFinal training data shape: {X_train.shape}")
    print(f"Final test data shape: {X_test.shape}")
    print(f"Number of unique classes in training: {len(np.unique(y_train))}")
        
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, num_classes):
    """訓練XGBoost模型"""
    print("\nTraining XGBoost model...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of classes: {num_classes}")
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        random_state=42,
        tree_method='hist',
        device='cuda',
        max_delta_step=1,
        min_child_weight=1,
        verbosity=2
    )
    
    print("Starting model training...")
    try:
        model.fit(
            X_train, 
            y_train,
            verbose=True
        )
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """評估模型"""
    print("\nEvaluating model...")
    global TARGET_MAPPING
    
    # 定義轉換函數
    def map_predictions(preds):
        return np.array([TARGET_MAPPING['new_to_original'].get(p, p) for p in preds])
    
    # 預測
    print("Making predictions...")
    y_pred = map_predictions(model.predict(X_test))
    y_pred_proba = model.predict_proba(X_test)
    
    # 生成分類報告
    print("\nGenerating classification report...")
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    return {
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def save_model(model, metrics, save_dir='base_models/xgboost_22cls'):
    """保存模型和相關信息"""
    print(f"\nSaving model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, 'model.json')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # 將numpy類型轉換為Python原生類型
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {str(key) if isinstance(key, np.integer) else key: convert_to_native_types(value) 
                    for key, value in obj.items()}
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
    
    # 保存全局映射
    global TARGET_MAPPING
    
    # 保存配置信息
    config = {
        'target_mapping': TARGET_MAPPING,
        'metrics': convert_to_native_types(metrics),
        'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_version': '1.0'
    }
    
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")

def main():
    print("=== Starting XGBoost Training for 22 Categories ===")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 載入數據
        features_data, train_indices, test_indices, label_encoder = load_data()
        
        # 準備訓練數據
        X_train, X_test, y_train, y_test = prepare_targeted_data(
            features_data, train_indices, test_indices, label_encoder
        )
        
        # 訓練模型
        model = train_model(
            X_train, y_train, num_classes=22
        )
        
        # 評估模型
        metrics = evaluate_model(
            model, X_test, y_test, label_encoder
        )
        
        # 保存模型和結果
        save_model(model, metrics)
        
        print("\n=== Training Summary ===")
        print("\nClassification Report:")
        print(json.dumps(metrics['classification_report'], indent=2))
        
        print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()