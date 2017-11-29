import os

def gpu():
    return os.getenv("GPU","no").lower() in ["1",1,"yes","true","t"]
