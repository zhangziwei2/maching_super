"""
FusionModel 类定义 - 供 train_fusion_model 和 chatter_diagnosis_skill 共用

决策融合系统，包含:
  - StandardScaler 标准化
  - 多个基分类器 (LightGBM, SVM, ExtraTrees, KNN, MLP, CatBoost)
  - StackingClassifier 元学习器
"""
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier


class FusionModel:
    """封装决策融合系统，便于序列化."""

    def __init__(self):
        self.scaler: StandardScaler = None
        self.classifiers: dict = {}
        self.f1_weights: dict = {}
        self.stacking: StackingClassifier = None

    def predict(self, X):
        return self.stacking.predict(X)

    def predict_proba(self, X):
        return self.stacking.predict_proba(X)
