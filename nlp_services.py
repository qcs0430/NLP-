"""
NLP服务模块 - 封装文本分类、情感分析、机器翻译功能
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import jieba
import os

# ===================== 文本分类服务 =====================

class TextClassifier:
    """文本分类服务 - 基于LSTM的中文新闻分类"""
    
    def __init__(self, vocab_path=None, model_path=None):
        self.vocab_path = vocab_path or '../nlp_models/cnews.vocab.txt'
        self.model_path = model_path or '../nlp_models/text_classifier.h5'
        self.categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        self.cat_to_id = dict(zip(self.categories, range(len(self.categories))))
        self.seq_length = 600
        self.model = None
        self.words = None
        self.word_to_id = None
        
    def _open_file(self, filename, mode='r'):
        return open(filename, mode, encoding='utf-8', errors='ignore')
        
    def _read_vocab(self):
        """读取词汇表"""
        if self.words is not None:
            return
        try:
            with self._open_file(self.vocab_path) as fp:
                self.words = [i.strip() for i in fp.readlines()]
            self.word_to_id = dict(zip(self.words, range(len(self.words))))
        except FileNotFoundError:
            self.words = []
            self.word_to_id = {}
            print(f"警告: 词汇表文件 {self.vocab_path} 不存在")
    
    def _build_model(self, vocab_size):
        """构建LSTM模型"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(vocab_size + 1, 128, input_length=self.seq_length))
        model.add(tf.keras.layers.LSTM(128))
        model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6, axis=1))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        return model
    
    def load_model(self):
        """加载模型"""
        self._read_vocab()
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                return True
            except Exception as e:
                print(f"加载模型失败: {e}")
        # 如果模型文件不存在，创建未训练的模型用于演示
        vocab_size = len(self.words) if self.words else 5000
        self.model = self._build_model(vocab_size)
        return False
    
    def predict(self, text):
        """预测文本分类"""
        if self.model is None:
            self.load_model()
        
        self._read_vocab()
        if not self.word_to_id:
            return {"category": "未知", "confidence": 0.0, "all_probs": {}}
        
        # 文本预处理
        content = list(text)
        data_id = [self.word_to_id.get(x, 0) for x in content]
        x_pad = keras.preprocessing.sequence.pad_sequences([data_id], self.seq_length)
        
        try:
            # 预测
            predictions = self.model.predict(x_pad, verbose=0)
            pred_idx = np.argmax(predictions[0])
            category = self.categories[pred_idx]
            confidence = float(predictions[0][pred_idx])
            
            # 所有类别的概率
            all_probs = {cat: float(predictions[0][i]) for i, cat in enumerate(self.categories)}
            
            return {
                "category": category,
                "confidence": confidence,
                "all_probs": all_probs
            }
        except Exception as e:
            return {"category": "预测失败", "confidence": 0.0, "error": str(e)}


# ===================== 情感分析服务 =====================

class SentimentAnalyzer:
    """情感分析服务 - 基于LSTM的中文情感分析"""
    
    def __init__(self, model_path=None, dict_path=None):
        self.model_path = model_path or '../nlp_models/sentiment_model.h5'
        self.dict_path = dict_path or '../nlp_models/sentiment_dict.npy'
        self.maxlen = 50
        self.model = None
        self.word_dict = None
        
    def _build_model(self, vocab_size):
        """构建LSTM模型"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Input
        
        model = Sequential()
        model.add(Input(shape=(self.maxlen,)))
        model.add(Embedding(vocab_size + 1, 256))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model
    
    def load_model(self):
        """加载模型"""
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                if os.path.exists(self.dict_path):
                    self.word_dict = np.load(self.dict_path, allow_pickle=True).item()
                return True
            except Exception as e:
                print(f"加载情感分析模型失败: {e}")
        
        # 创建示例模型用于演示
        self.model = self._build_model(50000)
        self.word_dict = {}
        return False
    
    def predict(self, text):
        """预测情感倾向"""
        if self.model is None:
            self.load_model()
        
        # 分词
        words = list(jieba.cut(text))
        
        # 如果没有词典，使用简单的规则判断
        if not self.word_dict:
            # 简单情感词典
            positive_words = ['好', '喜欢', '优秀', '棒', '赞', '开心', '快乐', '满意', '不错', '推荐', '优质', '漂亮', '舒服', '方便', '完美']
            negative_words = ['差', '烂', '糟糕', '坏', '讨厌', '失望', '难用', '垃圾', '恶心', '难过', '无聊', '后悔', '退货', '骗人']
            
            pos_count = sum(1 for w in words if any(pw in w for pw in positive_words))
            neg_count = sum(1 for w in words if any(nw in w for nw in negative_words))
            
            if pos_count > neg_count:
                sentiment = "正面"
                confidence = min(0.6 + pos_count * 0.1, 0.95)
            elif neg_count > pos_count:
                sentiment = "负面"
                confidence = min(0.6 + neg_count * 0.1, 0.95)
            else:
                sentiment = "中性"
                confidence = 0.5
                
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "positive_score": pos_count / max(len(words), 1),
                "negative_score": neg_count / max(len(words), 1)
            }
        
        # 使用模型预测
        try:
            word_ids = [self.word_dict.get(w, 0) for w in words]
            x_pad = keras.preprocessing.sequence.pad_sequences([word_ids], maxlen=self.maxlen)
            pred = self.model.predict(x_pad, verbose=0)[0][0]
            
            sentiment = "正面" if pred > 0.5 else "负面"
            confidence = float(pred) if pred > 0.5 else float(1 - pred)
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "positive_score": float(pred),
                "negative_score": float(1 - pred)
            }
        except Exception as e:
            return {"sentiment": "分析失败", "confidence": 0.0, "error": str(e)}


# ===================== 机器翻译服务 =====================

class MachineTranslator:
    """机器翻译服务 - 基于Seq2Seq的中英翻译"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or '../nlp_models/translation'
        self.encoder = None
        self.decoder = None
        self.inp_lang = None
        self.targ_lang = None
        self.max_length_inp = 20
        self.max_length_targ = 20
        self.units = 1024
        self.embedding_dim = 256
        
    def load_model(self):
        """加载翻译模型"""
        # 由于翻译模型比较复杂，这里提供简化版本
        # 在实际部署时可以加载训练好的模型
        return False
    
    def translate(self, text, direction='zh2en'):
        """翻译文本
        direction: 'zh2en' 中译英, 'en2zh' 英译中
        """
        # 简化实现 - 使用基础规则
        # 在实际应用中，应该加载训练好的Seq2Seq模型
        
        # 这里提供示例翻译（演示用）
        simple_dict_zh2en = {
            '你好': 'Hello',
            '谢谢': 'Thank you',
            '再见': 'Goodbye',
            '我': 'I',
            '你': 'You',
            '是': 'am/is/are',
            '爱': 'love',
            '喜欢': 'like',
            '中国': 'China',
            '学习': 'study',
            '工作': 'work',
            '朋友': 'friend',
            '今天': 'today',
            '明天': 'tomorrow',
            '早上好': 'Good morning',
            '晚安': 'Good night',
            '生病': 'sick',
            '打电话': 'call',
            '回家': 'go home',
        }
        
        simple_dict_en2zh = {v: k for k, v in simple_dict_zh2en.items()}
        
        if direction == 'zh2en':
            # 中译英
            words = list(jieba.cut(text))
            translated = []
            for word in words:
                if word in simple_dict_zh2en:
                    translated.append(simple_dict_zh2en[word])
                else:
                    translated.append(word)
            result = ' '.join(translated)
        else:
            # 英译中
            words = text.lower().split()
            translated = []
            for word in words:
                if word in simple_dict_en2zh:
                    translated.append(simple_dict_en2zh[word])
                else:
                    translated.append(word)
            result = ''.join(translated)
        
        return {
            "source": text,
            "translation": result,
            "direction": direction,
            "note": "演示翻译 - 建议使用豆包API获取更准确的翻译"
        }


# ===================== 综合分析服务 =====================

class MultiAnalyzer:
    """综合分析服务 - 整合多种NLP功能"""
    
    def __init__(self):
        self.classifier = TextClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.translator = MachineTranslator()
        
    def analyze(self, text):
        """综合分析文本"""
        results = {
            "input_text": text,
            "text_length": len(text),
            "word_count": len(list(jieba.cut(text)))
        }
        
        # 文本分类
        try:
            classification = self.classifier.predict(text)
            results["classification"] = classification
        except Exception as e:
            results["classification"] = {"error": str(e)}
        
        # 情感分析
        try:
            sentiment = self.sentiment_analyzer.predict(text)
            results["sentiment"] = sentiment
        except Exception as e:
            results["sentiment"] = {"error": str(e)}
        
        # 关键词提取（简单实现）
        words = list(jieba.cut(text))
        word_freq = {}
        for w in words:
            if len(w) > 1:  # 过滤单字
                word_freq[w] = word_freq.get(w, 0) + 1
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        results["keywords"] = [{"word": k, "count": v} for k, v in keywords]
        
        return results


# 创建全局服务实例
text_classifier = TextClassifier()
sentiment_analyzer = SentimentAnalyzer()
translator = MachineTranslator()
multi_analyzer = MultiAnalyzer()


def classify_text(text):
    """文本分类接口"""
    return text_classifier.predict(text)


def analyze_sentiment(text):
    """情感分析接口"""
    return sentiment_analyzer.predict(text)


def translate_text(text, direction='zh2en'):
    """翻译接口"""
    return translator.translate(text, direction)


def multi_analysis(text):
    """综合分析接口"""
    return multi_analyzer.analyze(text)
