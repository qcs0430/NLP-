"""
多功能智能问答系统 - 主应用入口
整合豆包API、文本分类、情感分析、机器翻译等多种NLP功能
"""

import os
import json
from jieba import lcut, add_word
import tensorflow as tf
from Seq2Seq import Encoder, Decoder
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

# 导入NLP服务模块
from nlp_services import (
    classify_text as nlp_classify,
    analyze_sentiment as nlp_sentiment,
    translate_text as nlp_translate,
    multi_analysis as nlp_multi_analysis
)

# 导入豆包API模块
from doubao_api import (
    doubao_chat,
    doubao_translate,
    doubao_classify,
    doubao_sentiment,
    doubao_analysis
)

# 导入DeepSeek API模块
from deepseek_api import deepseek_chat

# ===================== 用户管理 =====================
USERS_FILE = '../data/users.json'

def load_users():
    """加载用户数据"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    # 默认管理员账户
    return {'admin': {'password': '123456', 'role': 'admin', 'avatar': 'A'}}

def save_users(users):
    """保存用户数据"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

# 初始化用户数据
users_db = load_users()

# ===================== 聊天历史管理 =====================
CHATS_FILE = '../data/chats.json'

def load_all_chats():
    """加载所有用户的聊天记录"""
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_all_chats(chats):
    """保存所有用户的聊天记录"""
    os.makedirs(os.path.dirname(CHATS_FILE), exist_ok=True)
    with open(CHATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)

# 初始化聊天记录
chats_db = load_all_chats()

# ===================== 配置参数 =====================
# 代码11-18 调用Flask前端
# 设置参数
# 基于当前文件位置计算项目根目录，避免受运行目录影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../AI_QuestionAnswering/code
PROJECT_ROOT = os.path.dirname(BASE_DIR)              # .../AI_QuestionAnswering
data_path = os.path.join(PROJECT_ROOT, 'data', 'ids')  # 数据路径
embedding_dim = 256  # 词嵌入维度
hidden_dim = 512  # 隐层神经元个数
checkpoint_path = os.path.join(PROJECT_ROOT, 'tmp', 'model')  # 模型参数保存的路径
MAX_LENGTH = 50  # 句子的最大词长
CONST = {'_BOS': 0, '_EOS': 1, '_PAD': 2, '_UNK': 3}

# 聊天预测
def local_chat(sentence='你好'):
    # 初始化所有词语的哈希表
    table = tf.lookup.StaticHashTable(  # 初始化后即不可变的通用哈希表。
                initializer=tf.lookup.TextFileInitializer(
                    os.path.join(data_path, 'all_dict.txt'),
                    tf.string,
                    tf.lookup.TextFileIndex.WHOLE_LINE,
                    tf.int64,
                    tf.lookup.TextFileIndex.LINE_NUMBER
                ),  # 要使用的表初始化程序。有关支持的键和值类型，请参见HashTable内核。
                default_value=CONST['_UNK'] - len(CONST)  # 表中缺少键时使用的值。
            )

    # 实例化编码器和解码器
    encoder = Encoder(table.size().numpy() + len(CONST), embedding_dim, hidden_dim)
    decoder = Decoder(table.size().numpy() + len(CONST), embedding_dim, hidden_dim)
    optimizer = tf.keras.optimizers.Adam()  # 优化器
    # 模型保存路径
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    # 导入训练参数
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    # 给句子添加开始和结束标记
    sentence = '_BOS' + sentence + '_EOS'
    # 读取字段
    with open(os.path.join(data_path, 'all_dict.txt'), 'r', encoding='utf-8') as f:
        all_dict = f.read().split()
    # 构建: 词-->id的映射字典
    word2id = {j: i+len(CONST) for i, j in enumerate(all_dict)}
    word2id.update(CONST)
    # 构建: id-->词的映射字典
    id2word = dict(zip(word2id.values(), word2id.keys()))
    # 分词时保留_EOS 和 _BOS
    for i in ['_EOS', '_BOS']:
        add_word(i)
    # 添加识别不到的词，用_UNK表示
    inputs = [word2id.get(i, CONST['_UNK']) for i in lcut(sentence)]
    # 长度填充
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=MAX_LENGTH, padding='post', value=CONST['_PAD'])
    # 将数据转为tensorflow的数据类型
    inputs = tf.convert_to_tensor(inputs)
    # 空字符串，用于保留预测结果
    result = ''

    # 编码
    enc_out, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([word2id['_BOS']], 0)

    for t in range(MAX_LENGTH):
        # 解码
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        # 预测出词语对应的id
        predicted_id = tf.argmax(predictions[0]).numpy()
        # 通过字典的映射，用id寻找词，遇到_EOS停止输出
        if id2word.get(predicted_id, '_UNK') == '_EOS': 
            break
        # 未预测出来的词用_UNK替代
        result += id2word.get(predicted_id, '_UNK')
        dec_input = tf.expand_dims([predicted_id], 0)
    return result # 返回预测结果

# ===================== Flask应用 =====================
app = Flask(__name__, static_url_path='/static')
app.secret_key = 'nlp_question_answering_secret_key_2024'

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/admin')
def admin_page():
    """后台管理页面 - 需要admin登录"""
    return render_template('admin.html')

# ===================== 用户认证API =====================

@app.route('/api/login', methods=['POST'])
def api_login():
    """用户登录"""
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'success': False, 'message': '请输入用户名和密码'})
    
    if username in users_db and users_db[username]['password'] == password:
        role = users_db[username].get('role', 'user')
        return jsonify({
            'success': True, 
            'message': '登录成功',
            'user': {
                'username': username,
                'role': role,
                'avatar': users_db[username].get('avatar', username[0].upper())
            }
        })
    else:
        return jsonify({'success': False, 'message': '用户名或密码错误'})

@app.route('/api/register', methods=['POST'])
def api_register():
    """用户注册"""
    global users_db
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'success': False, 'message': '请输入用户名和密码'})
    
    if len(username) < 2:
        return jsonify({'success': False, 'message': '用户名至少2个字符'})
    
    if len(password) < 4:
        return jsonify({'success': False, 'message': '密码至少4个字符'})
    
    if username in users_db:
        return jsonify({'success': False, 'message': '用户名已存在'})
    
    users_db[username] = {
        'password': password,
        'role': 'user',
        'avatar': username[0].upper()
    }
    save_users(users_db)
    
    return jsonify({
        'success': True, 
        'message': '注册成功',
        'user': {
            'username': username,
            'role': 'user',
            'avatar': username[0].upper()
        }
    })

@app.route('/api/users', methods=['GET'])
def api_get_users():
    """获取所有用户列表"""
    user_list = []
    for username, info in users_db.items():
        user_list.append({
            'username': username,
            'role': info.get('role', 'user'),
            'avatar': info.get('avatar', username[0].upper()),
            'password': info.get('password', '')
        })
    return jsonify({'success': True, 'users': user_list})

@app.route('/api/users', methods=['POST'])
def api_add_user():
    """添加用户"""
    global users_db
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    role = data.get('role', 'user')
    
    if not username or not password:
        return jsonify({'success': False, 'message': '请输入用户名和密码'})
    
    if username in users_db:
        return jsonify({'success': False, 'message': '用户名已存在'})
    
    users_db[username] = {
        'password': password,
        'role': role,
        'avatar': username[0].upper()
    }
    save_users(users_db)
    
    return jsonify({'success': True, 'message': '用户添加成功'})

@app.route('/api/users/<username>', methods=['DELETE'])
def api_delete_user(username):
    """删除用户"""
    global users_db
    
    if username == 'admin':
        return jsonify({'success': False, 'message': '不能删除管理员账户'})
    
    if username not in users_db:
        return jsonify({'success': False, 'message': '用户不存在'})
    
    del users_db[username]
    save_users(users_db)
    
    return jsonify({'success': True, 'message': '用户删除成功'})

# ===================== 聊天历史API =====================

@app.route('/api/chats/<username>', methods=['GET'])
def api_get_user_chats(username):
    """获取用户的聊天记录"""
    if username not in chats_db:
        chats_db[username] = []
    return jsonify({'success': True, 'chats': chats_db[username]})

@app.route('/api/chats/<username>', methods=['POST'])
def api_save_user_chats(username):
    """保存用户的聊天记录"""
    global chats_db
    data = request.get_json() or {}
    chats = data.get('chats', [])
    
    chats_db[username] = chats
    save_all_chats(chats_db)
    
    return jsonify({'success': True, 'message': '聊天记录已保存'})

@app.route('/api/chats/<username>/<int:chat_id>', methods=['DELETE'])
def api_delete_chat(username, chat_id):
    """删除用户的某个聊天记录"""
    global chats_db
    
    if username in chats_db:
        chats_db[username] = [c for c in chats_db[username] if c.get('id') != chat_id]
        save_all_chats(chats_db)
    
    return jsonify({'success': True, 'message': '聊天记录已删除'})

@app.route('/message', methods=['POST'])
def reply():
    """统一消息处理接口"""
    req_msg = request.form.get('msg', '')
    mode = request.form.get('mode', 'doubao')  # 默认使用豆包
    
    if not req_msg.strip():
        return jsonify({'text': '请输入内容'})
    
    try:
        if mode == 'local':
            # 本地Seq2Seq聊天
            res_msg = local_chat(req_msg)
            res_msg = res_msg.replace('_UNK', '^_^').strip()
            if not res_msg:
                res_msg = '我们来聊聊天吧'
            return jsonify({'text': res_msg})
            
        elif mode == 'doubao':
            # 豆包智能对话
            result = doubao_chat(req_msg)
            if result['success']:
                return jsonify({'text': result['reply']})
            else:
                return jsonify({'text': f"豆包API暂时不可用: {result.get('reply', '未知错误')}"})
        
        elif mode == 'deepseek':
            # DeepSeek智能对话
            result = deepseek_chat(req_msg)
            if result['success']:
                return jsonify({'text': result['reply']})
            else:
                return jsonify({'text': f"DeepSeek API暂时不可用: {result.get('reply', '未知错误')}"})
                
        elif mode == 'classify':
            # 文本分类（优先使用豆包）
            result = doubao_classify(req_msg)
            if result['success']:
                response = f"📊 文本分类结果\n\n类别：{result['category']}"
            else:
                # 降级到本地模型
                result = nlp_classify(req_msg)
                response = f"📊 文本分类结果\n\n类别：{result['category']}\n置信度：{result['confidence']:.2%}"
            return jsonify({'text': response})
            
        elif mode == 'sentiment':
            # 情感分析（优先使用豆包）
            result = doubao_sentiment(req_msg)
            if result['success']:
                sentiment = result['sentiment']
                confidence = result.get('confidence', 0.8)
                emoji = '😊' if '正' in sentiment else ('😢' if '负' in sentiment else '😐')
                response = f"{emoji} 情感分析结果\n\n情感倾向：{sentiment}\n置信度：{confidence:.2%}"
            else:
                result = nlp_sentiment(req_msg)
                sentiment = result['sentiment']
                emoji = '😊' if sentiment == '正面' else ('😢' if sentiment == '负面' else '😐')
                response = f"{emoji} 情感分析结果\n\n情感倾向：{sentiment}\n置信度：{result['confidence']:.2%}"
            return jsonify({'text': response})
            
        elif mode == 'translate_zh2en':
            # 中译英
            result = doubao_translate(req_msg, "中文", "英文")
            if result['success']:
                response = f"🌐 翻译结果 (中→英)\n\n原文：{req_msg}\n译文：{result['translation']}"
            else:
                result = nlp_translate(req_msg, 'zh2en')
                response = f"🌐 翻译结果 (中→英)\n\n原文：{req_msg}\n译文：{result['translation']}\n\n注：{result.get('note', '')}"
            return jsonify({'text': response})
            
        elif mode == 'translate_en2zh':
            # 英译中
            result = doubao_translate(req_msg, "英文", "中文")
            if result['success']:
                response = f"🌐 翻译结果 (英→中)\n\n原文：{req_msg}\n译文：{result['translation']}"
            else:
                result = nlp_translate(req_msg, 'en2zh')
                response = f"🌐 翻译结果 (英→中)\n\n原文：{req_msg}\n译文：{result['translation']}\n\n注：{result.get('note', '')}"
            return jsonify({'text': response})
            
        elif mode == 'multi_analysis':
            # 综合分析
            result = doubao_analysis(req_msg)
            if result.get('success'):
                response = f"📋 综合分析报告\n\n"
                if 'category' in result:
                    response += f"📁 分类：{result['category']}\n"
                if 'sentiment' in result:
                    response += f"💭 情感：{result['sentiment']}\n"
                if 'keywords' in result:
                    keywords = result['keywords'] if isinstance(result['keywords'], list) else [result['keywords']]
                    response += f"🏷️ 关键词：{', '.join(keywords)}\n"
                if 'summary' in result:
                    response += f"📝 摘要：{result['summary']}"
                if 'analysis' in result:
                    response += f"\n{result['analysis']}"
            else:
                result = nlp_multi_analysis(req_msg)
                response = f"📋 综合分析报告\n\n"
                response += f"📏 文本长度：{result['text_length']} 字符\n"
                response += f"📊 词语数量：{result['word_count']} 个\n"
                if 'classification' in result:
                    response += f"📁 分类：{result['classification'].get('category', '未知')}\n"
                if 'sentiment' in result:
                    response += f"💭 情感：{result['sentiment'].get('sentiment', '未知')}\n"
                if 'keywords' in result:
                    kws = [k['word'] for k in result['keywords'][:5]]
                    response += f"🏷️ 关键词：{', '.join(kws)}"
            return jsonify({'text': response})
            
        else:
            return jsonify({'text': '未知的功能模式'})
            
    except Exception as e:
        return jsonify({'text': f'处理出错: {str(e)}'})

# ===================== 独立API端点 =====================

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """豆包对话API"""
    data = request.get_json() or {}
    message = data.get('message', request.form.get('msg', ''))
    if not message:
        return jsonify({'success': False, 'error': '消息不能为空'})
    result = doubao_chat(message)
    return jsonify(result)

@app.route('/api/deepseek', methods=['POST'])
def api_deepseek():
    """DeepSeek对话API"""
    data = request.get_json() or {}
    message = data.get('message', request.form.get('msg', ''))
    if not message:
        return jsonify({'success': False, 'error': '消息不能为空'})
    result = deepseek_chat(message)
    return jsonify(result)

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """文本分类API"""
    data = request.get_json() or {}
    text = data.get('text', request.form.get('text', ''))
    if not text:
        return jsonify({'success': False, 'error': '文本不能为空'})
    result = doubao_classify(text)
    return jsonify(result)

@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    """情感分析API"""
    data = request.get_json() or {}
    text = data.get('text', request.form.get('text', ''))
    if not text:
        return jsonify({'success': False, 'error': '文本不能为空'})
    result = doubao_sentiment(text)
    return jsonify(result)

@app.route('/api/translate', methods=['POST'])
def api_translate():
    """翻译API"""
    data = request.get_json() or {}
    text = data.get('text', request.form.get('text', ''))
    source = data.get('source_lang', '中文')
    target = data.get('target_lang', '英文')
    if not text:
        return jsonify({'success': False, 'error': '文本不能为空'})
    result = doubao_translate(text, source, target)
    return jsonify(result)

@app.route('/api/analysis', methods=['POST'])
def api_analysis():
    """综合分析API"""
    data = request.get_json() or {}
    text = data.get('text', request.form.get('text', ''))
    if not text:
        return jsonify({'success': False, 'error': '文本不能为空'})
    result = doubao_analysis(text)
    return jsonify(result)

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'services': {
            'doubao_api': True,
            'text_classification': True,
            'sentiment_analysis': True,
            'translation': True,
            'local_chat': True
        }
    })

# ===================== 启动应用 =====================
if __name__ == '__main__':
    print("=" * 50)
    print("多功能智能问答系统启动中...")
    print("=" * 50)
    print("功能列表:")
    print("  1. 豆包智能对话 - 基于大模型的智能问答")
    print("  2. 本地聊天 - 基于Seq2Seq的本地聊天")
    print("  3. 文本分类 - 新闻文本自动分类")
    print("  4. 情感分析 - 文本情感倾向分析")
    print("  5. 中英翻译 - 双向翻译服务")
    print("  6. 综合分析 - 多维度文本分析")
    print("=" * 50)
    print("访问地址: http://127.0.0.1:8808")
    print("=" * 50)
    app.run(host='127.0.0.1', port=8808, debug=False)
