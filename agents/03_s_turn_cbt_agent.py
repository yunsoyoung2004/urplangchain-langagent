
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def EmpathyAgent(user_input, emotion):
    return empathy_chain.run({"user_input": user_input, "emotion": emotion})

cbt_templates = {
    "evidence": PromptTemplate(
        input_variables=["user_input"],
        template="""
        사용자 발화: "{user_input}"

        아래 질문을 생성하세요:
        - 이 생각을 뒷받침하는 증거는 무엇인가요?
        - 반대되는 증거는 무엇인가요?
        - 주관적 느낌과 객관적 사실을 구분할 수 있도록 도와주세요.

        질문:
        """
    ),
    "alternative": PromptTemplate(
        input_variables=["user_input"],
        template="""
        사용자 발화: "{user_input}"

        이 상황에 대해 다른 설명이나 관점이 있을 수 있음을 유도하세요.
        - 인지적 유연성을 높이는 탐색적 질문을 하세요.

        질문:
        """
    ),
    "catastrophizing": PromptTemplate(
        input_variables=["user_input"],
        template="""
        사용자 발화: "{user_input}"

        아래 내용을 포함한 질문을 생성하세요:
        - 최악의 경우는 무엇인가요?
        - 그 상황에서의 대처 방법은?
        - 최선의 경우는?
        - 가장 현실적인 결과는?

        질문:
        """
    ),
    "impact": PromptTemplate(
        input_variables=["user_input"],
        template="""
        사용자 발화: "{user_input}"

        - 이 자동적 사고가 감정과 행동에 어떤 영향을 미치는지 탐색하세요.
        - 사고 전환을 통해 변화 가능성을 유도하세요.

        질문:
        """
    ),
    "third_person": PromptTemplate(
        input_variables=["user_input"],
        template="""
        사용자 발화: "{user_input}"

        아래 질문을 생성하세요:
        - 사랑하는 사람이 같은 상황이라면 뭐라고 말해주겠나요?
        - 자기연민을 유도하는 질문으로 바꿔보세요.

        질문:
        """
    ),
    "action_plan": PromptTemplate(
        input_variables=["user_input"],
        template="""
        사용자 발화: "{user_input}"

        지금 이 감정을 줄이기 위한 현실적인 행동 전략을 제시하도록 유도하세요.
        - 즉각적인 실천 가능한 조언을 얻는 질문을 하세요.

        질문:
        """
    )
}

class CBTAgent:
    def __init__(self, llm, user_input):
        self.llm = llm
        self.user_input = user_input
        self.question_types = list(cbt_templates.keys())

    def ask(self):
        for qtype in self.question_types:
            chain = LLMChain(llm=self.llm, prompt=cbt_templates[qtype])
            response = chain.run({"user_input": self.user_input})
            print(f"[{qtype.upper()} 질문]\nQ: {response}\n")
            self.user_input += f"\n{response}"
        return self.user_input
