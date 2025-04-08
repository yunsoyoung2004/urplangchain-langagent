
# 🌸 Bllossom 변화단계 통합 챗봇 (LoRA + Frameworks)

## 개요
이 챗봇은 변화단계 이론 기반의 심리 상담 흐름에 따라 LoRA 방식으로 파인튜닝된 Bllossom-8B 모델을 사용하고, 다양한 프레임워크(LangChain, CrewAI, AutoGen, LangGraph, LlamaIndex)를 통합해 실시간 라우팅, 협업, 그래프 흐름, 사용자 기억 기능을 지원합니다.

## 주요 통합 요소
- **LangChain**: router 및 LLMChain 구성 (`router.py`)
- **CrewAI**: agent 역할 분화 및 실행 (`crewai_manager.py`)
- **AutoGen**: 단계 전이 자동화 및 멀티에이전트 협업 (`autogen_flow.py`)
- **LangGraph**: 유저 변화 흐름 그래프 정의 (`graph/flow_graph.py`)
- **LlamaIndex**: 세션 기반 기억 시스템 (`user_memory/index.py`)

## 실행 방법
```bash
pip install -r requirements.txt
python main.py
```

## 포함된 데이터셋
- MI Dataset: `MI Dataset.csv`, `AnnoMI-simple.csv`, `AnnoMI-full.csv`
- PPI Dataset: `test.csv`, `dev.csv`

## 포함된 이미지
- `bllossom_chatbot_framework_chart.png`: 시스템 구조도 (논문용)
