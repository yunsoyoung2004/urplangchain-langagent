
from agents.router import route_user_input

def main():
    print("🌱 Bllossom 변화단계 챗봇에 오신 것을 환영합니다!")
    while True:
        user_input = input("\n사용자: ")
        if user_input.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다. 감사합니다!")
            break
        response = route_user_input(user_input)
        print(f"\n🌼 챗봇: {response}")
