class User:
    def __init__(self, id=None, username=None, password=None, is_banned=None,role=None):
        self.id = id
        self.username = username
        self.password = password
        self.is_banned = is_banned
        self.role = role