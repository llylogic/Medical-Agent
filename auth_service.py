# auth_service.py
import sqlite3
import uuid
from db_service import get_db_connection

def verify_login(username, password):
    """验证登录，成功则返回隐藏的 user_id 和 姓名"""
    if not username or not password:
        return False, None, None, "❌ 账号或密码不能为空！"
        
    conn = get_db_connection()
    cursor = conn.cursor()
    # 联合查询：验证密码的同时，取出患者档案里的真实姓名
    cursor.execute("""
        SELECT a.user_id, p.name 
        FROM Accounts a 
        JOIN Patients p ON a.user_id = p.user_id 
        WHERE a.username = ? AND a.password = ?
    """, (username, password))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, user[0], user[1], f"✅ 登录成功！欢迎回来，{user[1]}。"
    else:
        return False, None, None, "❌ 账号或密码错误，请重试。"

def register_account(username, password, name, gender, age):
    """注册新患者账号，自动分配系统级 User_ID"""
    if not all([username, password, name, age]):
        return "❌ 请填写完整的注册信息！"
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. 检查账号是否被占用
    cursor.execute("SELECT username FROM Accounts WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return "❌ 注册失败：该账号已被注册！"
        
    try:
        # 2. 生成隐秘的系统全局唯一 ID (如: U_a1b2c3d4)
        new_user_id = "U_" + uuid.uuid4().hex[:8]
        
        # 3. 开启事务：同时写入档案表和账号表
        cursor.execute("INSERT INTO Patients (user_id, name, gender, age) VALUES (?, ?, ?, ?)", 
                       (new_user_id, name, gender, int(age)))
        cursor.execute("INSERT INTO Accounts (username, password, user_id) VALUES (?, ?, ?)", 
                       (username, password, new_user_id))
        conn.commit()
        msg = f"🎉 注册成功！欢迎您，{name}。请切换到登录页面进行登录。"
    except Exception as e:
        conn.rollback()
        msg = f"❌ 系统异常，注册失败: {str(e)}"
    finally:
        conn.close()
        
    return msg