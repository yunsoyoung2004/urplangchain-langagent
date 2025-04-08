
# LlamaIndex 기반 유저 세션 임베딩 저장 구조 예시
user_memory = {}

def store_session(user_id, message):
    user_memory.setdefault(user_id, []).append(message)

def get_user_history(user_id):
    return user_memory.get(user_id, [])
