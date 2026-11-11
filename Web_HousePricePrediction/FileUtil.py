##copy trang 11
import pickle


class FileUtil:
    """Lớp tiện ích lưu / đọc mô hình bằng pickle."""

    @staticmethod
    def savemodel(model, filename):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            return True
        except Exception as e:
            print("An exception occurred:", e)
            return False

    @staticmethod
    def loadmodel(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print("An exception occurred:", e)
            return None
