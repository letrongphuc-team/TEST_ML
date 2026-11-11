import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dal.user_dao import UserDAO

uid="admin"
pwd="123"
us=UserDAO.login(uid,pwd)
if us is None:
    print("Login failed")
else:
    print("Login successfull")
    print("WELCOME=", us.username)