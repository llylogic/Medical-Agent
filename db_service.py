# db_service.py
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'hospital.db')

def init_sqlite_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 建表
    cursor.execute('''CREATE TABLE IF NOT EXISTS Patients (user_id TEXT PRIMARY KEY, name TEXT, gender TEXT, age INT, blood_type TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS Accounts (username TEXT PRIMARY KEY, password TEXT, user_id TEXT, FOREIGN KEY (user_id) REFERENCES Patients(user_id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS Medical_Records (record_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, diagnosis TEXT, current_meds TEXT, allergies TEXT, visit_date TEXT, FOREIGN KEY (user_id) REFERENCES Patients(user_id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS Appointments (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, department TEXT, appointment_date TEXT, status TEXT DEFAULT '已预约', FOREIGN KEY (user_id) REFERENCES Patients(user_id))''')
    
    # ================= 注入海量真实的测试数据 =================
    cursor.execute("SELECT count(*) FROM Accounts")
    if cursor.fetchone()[0] == 0:
        # 1. 账号表 (5个用户)
        accounts = [
            ('lisi', '123', 'U_1001'), ('wangwu', '123', 'U_1002'), 
            ('zhaoliu', '123', 'U_1003'), ('sunqi', '123', 'U_1004'), ('zhouba', '123', 'U_1005')
        ]
        cursor.executemany("INSERT INTO Accounts VALUES (?, ?, ?)", accounts)
        
        # 2. 患者基础表
        patients = [
            ('U_1001', '李四', '男', 30, 'O型'), ('U_1002', '王五', '女', 25, 'A型'),
            ('U_1003', '赵六', '男', 58, 'B型'), ('U_1004', '孙七', '女', 42, 'AB型'), ('U_1005', '周八', '男', 19, 'O型')
        ]
        cursor.executemany("INSERT INTO Patients VALUES (?, ?, ?, ?, ?)", patients)
        
        # 3. 电子病历表 (极其丰富的病史与用药)
        records = [
            ('U_1001', '重度抑郁症', '百忧解(氟西汀)', '青霉素过敏', '2025-10-01'),
            ('U_1002', '范科尼综合征', '氢氯噻嗪片', '无', '2025-11-15'),
            ('U_1003', '高血压3级、冠心病', '阿司匹林、硝苯地平', '磺胺类药物过敏', '2025-12-01'),
            ('U_1004', '支气管哮喘', '沙丁胺醇气雾剂', '花粉、海鲜过敏', '2026-01-10'),
            ('U_1005', '小儿进行性骨化性肌炎', '暂无', '头孢类过敏', '2026-02-20')
        ]
        cursor.executemany("INSERT INTO Medical_Records (user_id, diagnosis, current_meds, allergies, visit_date) VALUES (?, ?, ?, ?, ?)", records)
        
        # 4. 历史挂号表
        appointments = [
            ('U_1001', '精神卫生科', '2025-10-01 09:00', '已完成'),
            ('U_1003', '心血管内科', '2026-01-15 14:00', '已完成')
        ]
        cursor.executemany("INSERT INTO Appointments (user_id, department, appointment_date, status) VALUES (?, ?, ?, ?)", appointments)
        
        conn.commit()
    conn.close()

def get_db_connection():
    return sqlite3.connect(DB_PATH)