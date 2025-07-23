

# **Qwen Vision-Language 모델 제품군 종합 분석: 모델 선택 및 구현 가이드**

## **서론**

알리바바 클라우드가 개발한 Qwen Vision-Language (VL) 모델 제품군은 텍스트와 시각 정보를 통합적으로 이해하고 처리하는 멀티모달 AI 분야에서 중요한 위치를 차지하고 있습니다. 단순한 이미지 캡셔닝을 넘어 복잡한 문서 분석, 동영상 이해, 심지어 디지털 에이전트로서의 역할까지 수행하며 빠르게 발전하고 있습니다. 하지만 Qwen-VL, Qwen2-VL, Qwen2.5-VL 등 세대를 거듭하며 출시된 다양한 오픈소스 및 상용 모델들은 각기 다른 아키텍처, 성능, 라이선스, 그리고 배포 요구사항을 가지고 있어 사용자가 자신의 목적에 맞는 최적의 모델을 선택하기 어렵게 만듭니다.

본 보고서는 Qwen-VL 모델 제품군 전체에 대한 심층적이고 체계적인 분석을 제공하여, AI 개발자, 머신러닝 엔지니어, 연구자들이 정보에 입각한 기술적 결정을 내릴 수 있도록 지원하는 것을 목표로 합니다. 보고서는 Qwen-VL의 아키텍처 진화 과정부터 시작하여, 각 모델의 상세한 사양과 성능을 정량적으로 비교 분석합니다. 나아가 자체 호스팅을 위한 하드웨어 요구사항, 추론 최적화 기법, 그리고 API 사용과의 총소유비용(TCO) 비교 등 실질적인 배포 전략을 다룹니다. 최종적으로, 이 모든 분석을 종합하여 특정 사용 사례에 따른 최적의 모델을 선택할 수 있는 명확한 의사결정 프레임워크를 제시할 것입니다.

---

## **1\. Qwen-VL의 아키텍처 진화 과정**

Qwen-VL 제품군의 기술적 발전은 단순한 성능 향상을 넘어, 멀티모달 정보를 처리하는 근본적인 방식의 변화를 보여줍니다. 초기 모델이 기존 대규모 언어 모델(LLM)에 시각 능력을 '접목'하는 방식이었다면, 최신 모델은 시각과 언어를 본질적으로 함께 이해하는 '네이티브 멀티모달' 시스템으로 진화했습니다. 이러한 아키텍처의 변화가 어떻게 기능적 도약으로 이어졌는지 세대별로 분석합니다.

### **1.1. 기반 구축: 초기 Qwen-VL 아키텍처 (2023년 8월)**

초기 Qwen-VL 모델은 당시의 일반적인 비전-언어 모델(VLM) 접근법을 따르면서도 몇 가지 중요한 개선점을 도입하여 제품군의 기술적 토대를 마련했습니다.

* **핵심 구성 요소:** 이 1세대 모델은 세 가지 주요 부분으로 구성된 아키텍처를 확립했습니다: (1) 기반이 되는 대규모 언어 모델(LLM)인 Qwen-7B, (2) 시각 정보를 인코딩하는 비전 인코더(OpenCLIP ViT-bigG), 그리고 (3) 두 모달리티를 연결하는 위치 인식 비전-언어 어댑터(Position-aware Vision-Language Adapter)입니다.1 이 구조는 사전 훈련된 LLM에 시각 인식 능력을 이식하는 형태로, 효율적인 멀티모달 능력 확보를 목표로 했습니다.  
* **주요 기능:** 초기 모델임에도 불구하고 Qwen-VL은 여러 선구적인 기능을 선보였습니다. 영어와 중국어를 모두 지원하는 다국어 능력, 여러 이미지를 대화 중간에 삽입하여 비교하거나 질문할 수 있는 멀티 이미지 인터리빙(Multi-image interleaved conversation) 기능이 대표적입니다.1 특히, 당시 일반적인 224x224 해상도보다 높은 448x448 해상도의 이미지를 처리하여 더 세밀한 시각 정보 인식이 가능했으며, 이는 텍스트가 포함된 문서 이미지 분석에 유리하게 작용했습니다.5 또한, 중국어 환경에서 처음으로 바운딩 박스(Bounding Box)를 이용한 객체 위치 특정(Grounding) 기능을 지원했습니다.1  
* **기술적 한계:** 1세대 아키텍처의 근본적인 한계는 시각 정보를 거칠게(coarse-grained) 처리한다는 점에 있었습니다. 이미지를 고정된 크기의 패치로 분할하고, 이를 어댑터를 통해 고정된 길이의 시퀀스로 압축하는 과정에서 원본 이미지의 풍부한 시각적 세부 정보가 손실될 수밖에 없었습니다.3 이는 LLM에 전달되는 시각 정보의 '대역폭'을 제한하는 병목 현상을 야기했으며, 복잡한 시각적 추론 능력의 발전을 제약하는 요인이 되었습니다.

### **1.2. 세대적 도약: Qwen2-VL의 핵심 혁신 (2024년 8월)**

Qwen2-VL은 1세대의 한계를 극복하기 위해 시각 정보를 처리하는 방식 자체를 근본적으로 재설계했습니다. 이는 단순한 업그레이드가 아닌 패러다임의 전환이었으며, Qwen 제품군이 진정한 네이티브 멀티모달 모델로 나아가는 결정적인 단계였습니다.

* **네이티브 동적 해상도 (Naive Dynamic Resolution):** 이 기술은 Qwen2-VL의 가장 중요한 혁신입니다. 모든 이미지를 고정된 크기로 강제 변환하는 대신, 다양한 크기와 종횡비의 이미지를 있는 그대로 입력받아 가변적인 수의 시각 토큰으로 변환합니다.6 이는 고해상도 이미지의 세부 정보를 손실 없이 보존할 수 있게 해주며, 인간이 시각 정보를 처리하는 방식과 더 유사한 접근법입니다.8 이로써 1세대 모델의 정보 병목 현상을 구조적으로 해결했습니다.  
* **멀티모달 회전 위치 임베딩 (Multimodal Rotary Position Embedding, M-ROPE):** M-ROPE는 위치 정보를 1차원 텍스트, 2차원 이미지, 3차원 비디오라는 각 모달리티의 특성에 맞게 분해하여 인코딩하는 기술입니다.6 이를 통해 모델은 텍스트의 순서, 이미지 내 객체의 공간적 위치, 비디오의 시간적 흐름을 본질적으로 이해할 수 있게 되었습니다. 이는 복잡한 동영상 내용을 이해하거나, 시각적 환경과 상호작용하는 에이전트 기능을 구현하기 위한 핵심적인 전제 조건입니다.

### **1.3. 최첨단 기술의 완성: Qwen2.5-VL 아키텍처 (2025년 1월)**

Qwen2.5-VL은 Qwen2-VL의 혁신을 더욱 발전시켜 현재의 최첨단(State-of-the-Art) 성능을 완성했습니다. 특히 동영상 처리 능력과 연산 효율성 측면에서 주목할 만한 진전을 이루었습니다.

* **동영상을 위한 절대 시간 인코딩 (Absolute Time Encoding for Video):** M-ROPE 개념을 확장하여, 동영상의 시간적 위치 정보를 프레임 순서가 아닌 '절대 시간'과 정렬시켰습니다.10 이 기술 덕분에 Qwen2.5-VL은 한 시간이 넘는 긴 동영상도 처리할 수 있으며, 특정 이벤트가 발생한 시점을 초 단위로 정확하게 집어내는(second-level event localization) 것이 가능해졌습니다. 이는 단순한 프레임 나열을 넘어 동영상의 시간적 역학 관계를 진정으로 이해하게 되었음을 의미합니다.  
* **비전 트랜스포머 내 윈도우 어텐션 (Window Attention in Vision Transformer, ViT):** 동적 해상도 기술은 높은 성능을 제공하지만, 이미지 크기에 따라 연산량이 기하급수적으로 증가할 수 있는 문제를 안고 있었습니다. Qwen2.5-VL은 비전 트랜스포머에 윈도우 어텐션을 도입하여 이 문제를 해결했습니다.11 어텐션 계산을 특정 '창(window)' 내로 제한함으로써, 전체 연산량을 이미지 패치 수에 선형적으로 비례하도록 줄였습니다. 이 덕분에 고해상도 원본 이미지를 효율적으로 처리하면서도 성능 저하를 막을 수 있었습니다.  
* **QwenVL HTML 형식:** 연구 논문, 웹페이지, 금융 서식 등 복잡한 문서의 레이아웃 정보를 HTML과 유사한 구조로 추출하는 독자적인 문서 파싱 형식입니다.13 텍스트, 표, 이미지 등 각 요소의 공간적 관계를 보존함으로써, 단순한 광학 문자 인식(OCR)을 넘어 문서의 구조적 의미까지 깊이 이해할 수 있게 되었습니다.

이러한 아키텍처의 진화는 Qwen-VL이 '입력으로서의 비전'을 처리하는 모델에서 '네이티브 멀티모달 추론'이 가능한 시스템으로 발전했음을 명확히 보여줍니다. 초기 모델이 이미지를 LLM이 이해할 수 있는 형태로 '번역'하고 압축하는 데 중점을 두었다면, Qwen2-VL의 동적 해상도와 Qwen2.5-VL의 절대 시간 인코딩 도입은 모델이 시각 세계를 언어적 추론 경로와 깊이 통합된 풍부한 다차원적 표상으로 구축하게 만들었습니다. 이는 단순한 기능 개선이 아닌, 모델이 정보를 인식하고 처리하는 방식에 대한 철학적 변화이며, 단순 캡셔닝에서 복잡한 에이전트 작업 6 및 장편 동영상 분석 11으로 기능이 비약적으로 발전할 수 있었던 근본적인 이유입니다.

---

## **2\. Qwen-VL 모델 스펙트럼 비교 분석**

Qwen-VL 제품군은 다양한 사용자의 요구에 부응하기 위해 여러 종류의 모델을 제공합니다. 커뮤니티 개발과 연구를 위한 오픈소스 모델부터 최고 성능을 보장하는 상용 API 모델까지, 생태계 전체를 명확히 이해하는 것은 최적의 모델을 선택하기 위한 첫걸음입니다.

### **2.1. 오픈소스 모델: 파라미터 크기별 심층 분석**

허용적인 라이선스(주로 Apache 2.0) 하에 배포되는 오픈소스 모델들은 사용자가 직접 다운로드하여 자체 환경에 구축하고, 필요에 따라 미세 조정(fine-tuning)할 수 있는 유연성을 제공합니다.6

* **Qwen2-VL (2B, 7B):** 2세대 아키텍처를 기반으로 하는 모델들로, 준수한 성능과 효율성 사이의 균형을 제공합니다. 일반적인 멀티모달 작업에 적합한 시작점으로 볼 수 있습니다.6  
* **Qwen2.5-VL (3B, 7B, 32B, 72B):** 현재 가장 진보된 기술이 적용된 최신 오픈소스 제품군입니다.11  
  * **3B & 7B:** 이전 세대인 Qwen2-VL의 동일 파라미터 모델 대비 상당한 성능 향상을 이루었으며, 특히 3B 모델은 이전 세대의 7B 모델을 능가하는 성능을 보여주어 경량화된 환경에서의 효율성을 극대화했습니다.13  
  * **32B (Qwen2.5-VL-32B-Instruct):** 이 모델은 고성능과 접근성 사이의 최적의 지점에 위치합니다. GPT-4급의 능력을 보이면서도 고사양 개인용 워크스테이션(예: 64GB RAM을 갖춘 시스템)에서 구동이 가능하여, 강력한 성능을 필요로 하지만 대규모 클러스터 구축은 부담스러운 사용자에게 매력적인 선택지입니다.14  
  * **72B:** 오픈소스 모델 중 최고의 성능을 자랑하며, 일부 벤치마크에서는 유수의 상용 모델과 필적하는 결과를 보여줍니다. 다만, 상용 이용에 제약이 있을 수 있는 'Qwen 라이선스'로 배포되어 사용 전 라이선스 확인이 필수적입니다.6  
* **양자화 버전 (AWQ, GPTQ):** 대부분의 오픈소스 모델은 메모리 사용량과 연산 요구사항을 줄인 양자화 버전을 함께 제공합니다.15 AWQ(Activation-aware Weight Quantization)나 GPTQ(Generative Pre-trained Transformer Quantization) 같은 기술을 통해 모델 크기를 대폭 줄여, 리소스가 제한된 하드웨어에서도 배포가 가능하도록 지원합니다.20

### **2.2. 상용 플래그십 모델: Qwen-VL-Plus와 Qwen-VL-Max**

이 모델들은 알리바바 클라우드 API를 통해서만 접근 가능한 비공개 상용 모델입니다. 미공개 최신 연구 결과와 대규모 학습 데이터가 적용되어 Qwen-VL 제품군 중 가장 뛰어난 성능을 보장합니다.1

* **Qwen-VL-Plus:** '향상된(Enhanced)' 모델로 포지셔닝되며, 특히 이미지의 세부 사항과 텍스트 인식 능력이 크게 강화되었습니다. 수백만 픽셀 이상의 초고해상도 이미지를 지원하여, 최고 수준의 성능과 비용 효율성 사이의 균형점을 제공합니다.1  
* **Qwen-VL-Max:** '가장 유능한(Most Capable)' 플래그십 모델로, 시각적 추론과 지시 사항 이행 능력이 한 단계 더 발전했습니다. 가장 복잡하고 까다로운 작업을 위해 설계되었으며, 여러 벤치마크에서 GPT-4V나 Gemini와 같은 경쟁 모델을 능가하는 성능을 보입니다. 특히 중국어 관련 작업에서는 더욱 두드러진 강점을 나타냅니다.1

알리바바의 모델 출시 전략은 개발자들을 유인하고 생태계를 구축하기 위한 '깔때기(Funnel)' 구조로 해석될 수 있습니다. 먼저, 강력한 성능의 오픈소스 모델들을 대거 공개하여 6 개발자 커뮤니티를 활성화하고 Qwen 아키텍처에 대한 친숙도를 높입니다. 개발자들은 이 무료 모델들을 활용하여 프로토타입을 만들고 기술을 검증합니다. 그러나 최고의 성능과 안정성이 요구되는 상용 서비스 단계에 이르면, 자연스럽게 가장 강력한 성능을 제공하는 상용 API 모델인 Qwen-VL-Max로의 전환을 고려하게 됩니다.1 72B 오픈소스 모델을 상업적 이용이 까다로운 라이선스로 배포하는 것 6 역시 이러한 전략의 일환으로, 최고 수준의 성능을 맛보게 하되 대규모 상용화를 위해서는 간편한 API를 선택하도록 유도하는 역할을 합니다.

### **표 1: Qwen-VL 모델 제품군 상세 사양 비교**

| 모델명 | 시리즈 | 파라미터 | 기반 LLM | 라이선스 | 제공 형태 | 주요 아키텍처 특징 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Qwen-VL** | Qwen-VL | 7B | Qwen-7B | Tongyi Qianwen LICENSE | Hugging Face, ModelScope | 고해상도(448x448) 입력, 위치 인식 어댑터 |
| **Qwen-VL-Chat** | Qwen-VL | 7B | Qwen-7B | Tongyi Qianwen LICENSE | Hugging Face, ModelScope | Qwen-VL의 대화형 미세 조정 버전 |
| **Qwen2-VL-2B-Instruct** | Qwen2-VL | 2B | Qwen2 | Apache 2.0 | Hugging Face, ModelScope | 동적 해상도, M-ROPE |
| **Qwen2-VL-7B-Instruct** | Qwen2-VL | 7B | Qwen2 | Apache 2.0 | Hugging Face, ModelScope | 동적 해상도, M-ROPE |
| **Qwen2.5-VL-3B-Instruct** | Qwen2.5-VL | 4B | Qwen2.5 | Apache 2.0 | Hugging Face, ModelScope | 절대 시간 인코딩, 윈도우 어텐션 |
| **Qwen2.5-VL-7B-Instruct** | Qwen2.5-VL | 8B | Qwen2.5 | Apache 2.0 | Hugging Face, ModelScope | 절대 시간 인코딩, 윈도우 어텐션 |
| **Qwen2.5-VL-32B-Instruct** | Qwen2.5-VL | 33B | Qwen2.5 | Apache 2.0 | Hugging Face, ModelScope | 절대 시간 인코딩, 윈도우 어텐션, RLHF |
| **Qwen2.5-VL-72B-Instruct** | Qwen2.5-VL | 73B | Qwen2.5 | Qwen License | Hugging Face, ModelScope | 절대 시간 인코딩, 윈도우 어텐션 |
| **Qwen-VL-Plus** | Qwen-VL | 비공개 | 비공개 | 상용 | Alibaba Cloud API | 초고해상도 이미지 지원, 텍스트 인식 강화 |
| **Qwen-VL-Max** | Qwen-VL | 비공개 | 비공개 | 상용 | Alibaba Cloud API | 시각적 추론 및 지시 이행 능력 극대화 |

---

## **3\. 정량적 성능 분석 및 벤치마킹**

Qwen-VL 모델의 성능을 객관적으로 평가하기 위해, 주요 멀티모달 벤치마크에서의 결과를 경쟁 모델들과 비교 분석합니다. 이 분석은 Qwen-VL 제품군이 특히 강점을 보이는 영역을 명확히 하고, 모델 선택의 정량적 근거를 제공합니다.

Qwen-VL 제품군이 특정 벤치마크에서 지속적으로 최상위 성능을 보이는 것은 우연이 아닙니다. **DocVQA**, **OCRBench**와 같은 문서 이해 벤치마크와 **AITZ**, **ScreenSpot**과 같은 에이전트 능력 평가 벤치마크에서의 강세는 7 알리바바가 범용적인 창의적 작업보다는 정확성과 신뢰성이 중요한 고부가가치 기업 자동화 시장을 전략적으로 목표하고 있음을 시사합니다. 고해상도 처리, HTML 파싱과 같은 아키텍처 혁신은 이러한 전략적 방향을 직접적으로 뒷받침합니다.

### **3.1. 문서 및 다이어그램 이해**

이 영역은 Qwen-VL 제품군의 가장 두드러진 강점 중 하나입니다. 복잡한 구조의 문서에서 정보를 정확히 인식하고 추출하는 능력은 기업 자동화의 핵심입니다.

* **주요 벤치마크:** **DocVQA** (문서 기반 질의응답), **InfoVQA** (인포그래픽 질의응답), **ChartQA** (차트 질의응답), **OCRBench** (광학 문자 인식 종합 평가) 등에서 평가된 성능을 분석합니다.7  
* **성능 분석:** 플래그십 모델인 Qwen-VL-Max와 Qwen2.5-VL-72B는 이 분야에서 지속적으로 최첨단(SOTA) 성능을 기록하며, 종종 GPT-4o나 Claude 3.5 Sonnet과 같은 강력한 경쟁 모델들을 능가합니다.7 예를 들어, DocVQA 벤치마크에서 Qwen2.5-VL-72B는 96.4점이라는 높은 점수를 기록하여, 현존하는 최고 수준의 모델들과 어깨를 나란히 합니다.25 이는 고해상도 입력 처리와 미세한 시각적 특징을 이해하도록 설계된 아키텍처의 직접적인 결과입니다.

### **3.2. 수학 및 논리적 추론**

시각 정보에 기반한 복잡한 추론 능력은 모델의 지능을 평가하는 중요한 척도입니다.

* **주요 벤치마크:** 시각적으로 제시된 수학 문제를 해결하는 **MathVista**와 대학 수준의 멀티모달 문제 해결 능력을 평가하는 **MMMU** 벤치마크를 중심으로 분석합니다.7  
* **성능 분석:** Qwen2.5-VL-72B는 MathVista에서 74.8점을 획득하며 경쟁 모델들을 앞서는 등 최상위권의 성능을 보여줍니다.25 특히, 강화학습을 통해 미세 조정된 최신 모델인 Qwen2.5-VL-32B-Instruct는 수학적 추론 능력이 더욱 향상되었다고 보고되었습니다.17 이는 Qwen-VL이 단순한 시각적 인식을 넘어, 다단계의 복잡한 논리적 추론을 수행하는 방향으로 발전하고 있음을 보여줍니다.

### **3.3. 장편 동영상 이해**

동영상은 시간적 차원이 추가된 복잡한 데이터 형태로, 이를 깊이 있게 이해하는 것은 차세대 멀티모달 기술의 핵심입니다.

* **주요 벤치마크:** **VideoMME**, **LVBench**, **CharadesSTA** 등 동영상 기반 질의응답 및 이벤트 시간 특정 벤치마크를 통해 성능을 평가합니다.25  
* **성능 분석:** M-ROPE와 절대 시간 인코딩과 같은 독자적인 아키텍처 혁신 덕분에 Qwen2.5-VL 제품군은 이 분야에서 뚜렷한 경쟁 우위를 가집니다. 72B 모델은 VideoMME에서 73.3점, CharadesSTA에서 50.9점을 기록하며 GPT-4o의 성능을 크게 상회합니다.25 이는 Qwen-VL의 특화된 동영상 처리 아키텍처가 효과적으로 작동하고 있음을 입증합니다.

### **3.4. 에이전트 기능 및 제어**

시각적 입력을 바탕으로 디지털 환경을 조작하고 도구를 사용하는 에이전트 기능은 멀티모달 AI의 궁극적인 활용 분야 중 하나입니다.

* **주요 벤치마크:** **AITZ** (Android in the Zoo), **ScreenSpot** 등 모바일 앱이나 컴퓨터 화면을 보고 작업을 수행하는 능력을 측정하는 벤치마크를 분석합니다.25  
* **성능 분석:** Qwen2.5-VL-72B는 AITZ 벤치마크에서 83.2점이라는 놀라운 점수를 기록했는데, 이는 GPT-4o의 35.3점과 비교해 압도적인 수치입니다.25 시각적 맥락을 이해하고 그에 따라 추론하며 도구를 사용하는 이 능력은 Qwen-VL 제품군의 핵심 차별점이며, 자동화 및 인간-컴퓨터 상호작용 분야에서의 미래 응용 가능성을 시사합니다.6

### **표 2: 주요 벤치마크 성능 비교 (Qwen-VL vs. SOTA 모델)**

| 벤치마크 | 측정 능력 | Qwen2.5-VL-72B | Qwen2.5-VL-32B | GPT-4o | Claude 3.5 Sonnet |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **DocVQA** | 문서 질의응답 | **96.4** | \- | 91.1 | 95.2 |
| **OCRBenchV2** | OCR 종합 | **61.5** | \- | 46.5 | 45.2 |
| **MathVista** | 수학 추론 | **74.8** | 74.8\* | 63.8 | 65.4 |
| **MMMU** | 대학 수준 추론 | 70.2 | 70.2\* | 70.3 | **70.4** |
| **VideoMME** | 동영상 이해 | **73.3** | \- | 71.9 | 60.0 |
| **CharadesSTA** | 동영상 시간 특정 | **50.9** | \- | 35.7 | \- |
| **AITZ** | 안드로이드 제어 | **83.2** | \- | 35.3 | \- |

주: Qwen2.5-VL-32B의 MathVista 및 MMMU 점수는 해당 모델의 발표 자료에서 인용되었으며 18, 다른 모델들과의 직접 비교 시점은 다를 수 있습니다. 모든 점수는 참고 자료 18에 기반합니다. 최고 점수는

**굵게** 표시되었습니다.

---

## **4\. 배포 및 구현을 위한 실용 가이드**

모델의 이론적 성능만큼이나 중요한 것은 실제 환경에서의 배포 및 운영 가능성입니다. 이 섹션에서는 자체 호스팅을 위한 하드웨어 요구사항부터 추론 성능을 극대화하기 위한 최적화 기법, 그리고 경제성 분석까지 실질적인 정보를 다룹니다.

### **4.1. 인프라 설계: 자체 호스팅을 위한 하드웨어 요구사항**

오픈소스 Qwen-VL 모델을 자체 서버에 배포하기 위해서는 모델의 크기와 정밀도(precision)에 맞는 충분한 하드웨어 사양을 갖추어야 합니다.

* **분석:** 여러 자료를 종합하여 각 모델 크기(3B, 7B, 32B, 72B)를 운영하는 데 필요한 GPU VRAM, 시스템 RAM, CPU 요구사항을 정리합니다.25  
* **모델별 요구사항 예시:**  
  * **3B 모델:** 추론을 위해 최소 8GB의 VRAM이 필요합니다 (예: NVIDIA RTX 3060).25  
  * **7B 모델:** 추론을 위해 최소 16GB에서 24GB의 VRAM이 권장됩니다 (예: RTX 3090, A4000).25 미세 조정을 위해서는 최소 32GB 이상의 VRAM을 갖춘 전문가용 GPU(예: A100, V100)가 필요합니다.35  
  * **32B 모델:** 추론을 위해 다중 GPU 구성(예: 2x A100 40GB) 또는 단일 고성능 GPU(예: H100 80GB)가 필요합니다.25  
  * **72B 모델:** 4x A100 80GB 또는 2x H100 80GB와 같이 상당한 규모의 GPU 클러스터가 필수적입니다.25

### **표 3: Qwen-VL 모델 자체 호스팅을 위한 GPU VRAM 요구사항**

| 모델 (파라미터) | 추론 (FP16) 최소/권장 | 추론 (4-bit 양자화) 최소/권장 | 미세 조정 (Fine-Tuning) 권장 |
| :---- | :---- | :---- | :---- |
| **Qwen2.5-VL-3B** | 8 GB / 16 GB | 4 GB / 8 GB | 18+ GB |
| **Qwen2.5-VL-7B** | 16 GB / 24 GB | 8 GB / 12 GB | 32+ GB |
| **Qwen2.5-VL-32B** | 48 GB / 80 GB | 24 GB / 32 GB | 80+ GB (다중 GPU) |
| **Qwen2.5-VL-72B** | 160 GB / 192 GB (다중 GPU) | 48 GB / 64 GB (다중 GPU) | 320+ GB (다중 GPU) |

주: 위 수치는 여러 자료 25를 종합한 추정치이며, 실제 요구사항은 배치 크기, 시퀀스 길이, 사용하는 프레임워크에 따라 달라질 수 있습니다.

### **4.2. 추론 최적화: 양자화 및 서빙 프레임워크 기술 검토**

대규모 모델을 실제 서비스에 적용하기 위해서는 추론 속도를 높이고 리소스 사용량을 줄이는 최적화 과정이 필수적입니다.

* **vLLM을 이용한 서빙:** Qwen 모델 배포 시 vLLM 프레임워크 사용이 강력히 권장됩니다. vLLM은 PagedAttention과 같은 기술을 통해 메모리를 효율적으로 관리하고, 연속적인 요청을 묶어 처리(continuous batching)하여 높은 처리량(throughput)을 달성합니다.38 또한, OpenAI와 호환되는 API 엔드포인트를 쉽게 구축할 수 있는 장점이 있습니다. 다만, 메모리 부족(OOM) 오류를 방지하기 위해  
  \--max-model-len(최대 시퀀스 길이)이나 \--gpu-memory-utilization(GPU 메모리 사용률)과 같은 옵션을 적절히 조절하는 노하우가 필요합니다.38  
* **양자화 (AWQ vs. GPTQ):** 모델의 가중치를 낮은 정밀도(예: 4-bit)로 변환하여 크기를 줄이는 기술입니다.  
  * **AWQ (Activation-aware Weight Quantization):** 모델의 성능에 중요한 영향을 미치는 가중치를 식별하여 보호하고, 덜 중요한 가중치를 더 압축하는 방식으로 성능 저하를 최소화합니다.40 vLLM에서 잘 지원됩니다.41  
  * **GPTQ (Generative Pre-trained Transformer Quantization):** 훈련 후 양자화(Post-Training Quantization) 기법으로, ExLlamaV2와 같은 최적화된 커널과 함께 사용할 경우 더 빠른 추론 속도를 제공할 수 있습니다.40  
  * **성능 영향:** 양자화는 VRAM 사용량을 최대 75%까지 획기적으로 줄일 수 있지만, 필연적으로 약간의 성능 저하를 동반합니다.43 특히 복잡한 추론이나 섬세한 작업에서는 이러한 성능 저하가 두드러질 수 있습니다. 한 사용자는 양자화 모델이 원본 모델과 달리 운전 결정과 같은 미묘한 작업에서 3%의 추가적인 오류를 보였다고 보고했습니다.45 이는 양자화가 단순한 기술적 최적화를 넘어, 서비스의 신뢰도에 영향을 미칠 수 있는 중요한 결정임을 시사합니다. 따라서 미션 크리티컬한 애플리케이션에서는 양자화로 인한 비용 절감 효과와 잠재적인 성능 저하의 위험을 신중하게 평가해야 합니다.

### **4.3. 경제성 분석: 자체 호스팅과 API 접근의 총소유비용(TCO) 비교**

모델 사용 방식은 기술적 선택일 뿐만 아니라 경제적 결정이기도 합니다.

* **API 가격 정책:** Qwen-VL-Plus와 Qwen-VL-Max는 알리바바 클라우드를 통해 토큰 사용량 기반으로 과금됩니다. 특히 최근 대대적인 가격 인하를 단행하여 매우 경쟁력 있는 가격을 제공합니다.21 예를 들어, Qwen-VL-Max는 85% 가격이 인하되어 입력 토큰 1,000개당 0.003위안(약 $0.00041 USD) 수준입니다.14  
* **자체 호스팅 비용:** 자체 호스팅은 GPU 구매를 위한 높은 초기 투자 비용(예: 8x H100 서버 구성에 $300,000) 외에도 전력, 냉각, 상면 비용, 유지보수를 위한 전문 인력 등 상당한 운영 비용이 지속적으로 발생합니다.49  
* **손익분기점 분석:** 일반적으로 API를 사용하는 것이 초기 비용 부담이 없고 관리가 용이하여 대부분의 경우 더 경제적입니다. 자체 호스팅이 경제성을 갖기 시작하는 시점은 월간 토큰 사용량이 매우 많고(예: 1억 토큰 이상) 사용 패턴이 예측 가능할 때입니다.49 데이터 프라이버시나 보안 규제로 인해 외부 API 사용이 불가능한 경우에도 자체 호스팅이 유일한 대안이 될 수 있습니다.

---

## **5\. 최적의 모델 선택을 위한 의사결정 프레임워크**

지금까지의 분석을 종합하여, 사용자의 구체적인 요구사항과 제약 조건에 따라 최적의 Qwen-VL 모델을 선택할 수 있는 명확하고 실행 가능한 프레임워크를 제시합니다.

### **5.1. 사용 사례 기반 추천 모델**

* **시나리오 1: 최고 성능이 요구되는 기업용 및 복잡한 추론 작업**  
  * **추천 모델:** **Qwen-VL-Max API**  
  * **근거:** 문서 분석, 금융 보고, 의료 영상 판독 등 최고의 정확성과 신뢰성이 요구되는 미션 크리티컬한 작업에는 상용 플래그십 모델이 가장 적합합니다. 복잡한 인프라 관리 부담 없이 최고의 성능을 즉시 활용할 수 있으며 1, 경쟁력 있는 API 가격 정책 덕분에 많은 기업 애플리케이션에서 경제적으로도 합리적인 선택이 될 수 있습니다.47  
* **시나리오 2: 데이터 통제권이 중요한 고성능 자체 호스팅**  
  * **추천 모델:** **Qwen2.5-VL-72B** 또는 **Qwen2.5-VL-32B**  
  * **근거:** 데이터 프라이버시, 보안 규정 준수, 또는 독자적인 데이터로 깊이 있는 미세 조정이 필요한 경우 자체 호스팅이 필수적입니다. 72B 모델은 상용 모델에 필적하는 성능을 제공하지만, 막대한 하드웨어 투자가 필요합니다.25 32B 모델은 비교적 접근 가능한 하드웨어에서 최상위권에 근접한 성능을 제공하여, 예산과 성능 요구사항 사이에서 뛰어난 균형을 이룹니다.17  
* **시나리오 3: 범용 멀티모달 애플리케이션을 위한 균형 잡힌 선택**  
  * **추천 모델:** **Qwen2.5-VL-7B**  
  * **근거:** 이 모델은 오픈소스 제품군 중 가장 실용적인 '일꾼(workhorse)'입니다. 단일 고사양 GPU(예: RTX 4090)에서 구동 가능하며 25, 다양한 작업에서 뛰어난 성능을 보여줍니다. 허용적인 Apache 2.0 라이선스는 상용 제품 개발에 적합하며, 대규모 클러스터 구축 비용 없이도 모델에 대한 완전한 통제권을 확보하고자 할 때 최적의 선택입니다.  
* **시나리오 4: 리소스가 제한된 환경 및 엣지 AI 배포**  
  * **추천 모델:** **Qwen2.5-VL-3B (양자화 버전)**  
  * **근거:** 모바일 기기나 엣지 서버와 같이 계산 자원이 제한된 환경을 위한 모델입니다. 3B 모델은 이전 세대의 7B 모델보다도 우수한 성능을 보여주면서도 13, 양자화를 통해 VRAM 요구사항을 더욱 낮출 수 있습니다. 응답 지연 시간(latency)과 리소스 사용량이 가장 중요한 제약 조건일 때 가장 효율적인 선택입니다.

### **5.2. 알려진 한계, 위험 및 윤리적 고려사항**

* **미세 조정의 어려움:** 커뮤니티의 논의에 따르면, Qwen-VL 모델의 미세 조정은 몇 가지 어려움을 동반합니다. 특정 데이터셋에 과적합(overfitting)되어 범용 추론 능력을 잃어버리는 '치명적 망각(catastrophic forgetting)' 현상이 보고되었습니다.52 예를 들어, 특정 문서 형식에 맞춰 미세 조정한 모델이 이미지 분석 능력 자체를 잃어버리는 경우가 있었습니다.52 이는 미세 조정이 단순한 성능 향상 수단이 아니라, 범용 모델을 특정 작업에 특화된 '전문가'로 바꾸는 전략적 결정임을 의미합니다. 따라서 전문화로 얻는 이득이 유연성 상실의 비용보다 큰지 신중히 판단해야 합니다.  
* **추론 안정성 및 버그:** 특정 버전의 vLLM과 같은 서빙 프레임워크에서 이미지 인식 성능이 저하되는 등, 소프트웨어 환경에 따른 안정성 문제가 보고된 바 있습니다.53 성공적인 배포를 위해서는 신중한 환경 구성과 충분한 테스트가 요구됩니다.  
* **편향 및 윤리적 위험:** Qwen 모델은 학습 데이터 단계에서 유해 콘텐츠를 필터링하는 과정을 거치지만 55, 웹 스케일의 데이터를 학습한 모든 LLM과 마찬가지로 사회에 내재된 성별, 인종, 문화적 편향을 학습하고 재현할 위험이 있습니다.56 또한, 더 긴 답변이나 비교 시 먼저 제시된 답변을 선호하는 것과 같은 인지적 편향을 보일 수도 있습니다.56 따라서 대중에게 공개되는 애플리케이션에 적용할 경우, 자체적인 편향성 및 공정성 감사를 수행하고 잠재적인 위험을 완화하기 위한 장치를 마련하는 것이 필수적입니다.

## **결론 및 최종 제언**

Qwen-VL 모델 제품군은 알리바바 클라우드의 명확한 전략 아래 빠르게 발전해 온 강력한 멀티모달 AI 솔루션입니다. 초기 모델부터 최신 Qwen2.5-VL에 이르기까지의 아키텍처 진화는 단순한 성능 개선을 넘어, 시각 정보를 보다 근본적이고 인간과 유사한 방식으로 처리하려는 일관된 목표를 보여줍니다. 특히 동적 해상도, M-ROPE, 절대 시간 인코딩과 같은 혁신 기술들은 Qwen-VL이 문서 분석, 동영상 이해, 에이전트 기능과 같은 고부가가치 기업 자동화 영역에서 세계 최고 수준의 경쟁력을 갖추게 한 원동력입니다.

사용자 입장에서 최적의 모델을 선택하는 것은 여러 차원의 트레이드오프를 고려해야 하는 복합적인 결정 과정입니다.

1. **성능과 비용 (API vs. 자체 호스팅):** 최고의 성능과 관리 편의성을 원한다면 Qwen-VL-Max API가 최선의 선택입니다. 반면, 데이터 통제권이 중요하고 월간 수억 토큰 이상의 대규모 트래픽이 예상된다면, 높은 초기 비용을 감수하고 자체 호스팅을 고려할 수 있습니다.  
2. **성능과 접근성 (대형 vs. 소형 모델):** 72B 모델은 최상의 오픈소스 성능을 제공하지만 막대한 인프라를 요구합니다. 32B 모델은 성능과 접근성 사이의 뛰어난 균형점을, 7B 모델은 단일 GPU 환경에서의 실용적인 고성능을, 3B 모델은 엣지 환경에서의 효율성을 대표합니다.  
3. **정확성과 효율성 (원본 vs. 양자화):** 양자화는 배포 비용을 획기적으로 낮추지만, 특히 섬세하고 중요한 작업에서 신뢰도 저하라는 숨겨진 비용을 초래할 수 있습니다.  
4. **전문화와 범용성 (미세 조정 vs. 프롬프트 엔지니어링):** 미세 조정은 특정 작업의 성능을 극대화하지만, 모델의 범용성을 해칠 위험이 있습니다. 다양한 작업을 처리해야 한다면, 미세 조정 대신 정교한 프롬프트 엔지니어링 전략을 사용하는 것이 더 효과적일 수 있습니다.

궁극적으로, Qwen-VL 제품군 내에서의 선택은 기술적 사양뿐만 아니라, 프로젝트의 예산, 데이터 보안 정책, 요구되는 성능 수준, 그리고 장기적인 운영 전략과 같은 비즈니스 목표와 깊이 연관되어 있습니다. 본 보고서에서 제시된 다각적인 분석과 의사결정 프레임워크가 사용자의 복잡한 요구사항에 가장 부합하는 최적의 Qwen-VL 모델을 성공적으로 선택하고 구현하는 데 있어 신뢰할 수 있는 가이드가 되기를 바랍니다.

#### **참고 자료**

1. The official repo of Qwen-VL (通义千问-VL) chat & pretrained large vision language model proposed by Alibaba Cloud. \- GitHub, 7월 23, 2025에 액세스, [https://github.com/QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL)  
2. Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities \- arXiv, 7월 23, 2025에 액세스, [https://arxiv.org/pdf/2308.12966v1.pdf?ref=blog.runpod.io](https://arxiv.org/pdf/2308.12966v1.pdf?ref=blog.runpod.io)  
3. Large-scale Vision Language Models (LVLMs): Qwen-VL and Qwen-VL-Chat \- Encord, 7월 23, 2025에 액세스, [https://encord.com/blog/qwen-vl-large-scale-vision-language-models/](https://encord.com/blog/qwen-vl-large-scale-vision-language-models/)  
4. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond \- arXiv, 7월 23, 2025에 액세스, [http://arxiv.org/pdf/2308.12966](http://arxiv.org/pdf/2308.12966)  
5. cognitedata/Qwen-VL-finetune: The official repo of Qwen-VL (通义千问-VL) chat & pretrained large vision language model proposed by Alibaba Cloud. \- GitHub, 7월 23, 2025에 액세스, [https://github.com/cognitedata/Qwen-VL-finetune](https://github.com/cognitedata/Qwen-VL-finetune)  
6. xwjim/Qwen2-VL: Qwen2-VL is the multimodal large ... \- GitHub, 7월 23, 2025에 액세스, [https://github.com/xwjim/Qwen2-VL](https://github.com/xwjim/Qwen2-VL)  
7. Qwen2-VL: To See the World More Clearly | Qwen, 7월 23, 2025에 액세스, [https://qwenlm.github.io/blog/qwen2-vl/](https://qwenlm.github.io/blog/qwen2-vl/)  
8. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution \- arXiv, 7월 23, 2025에 액세스, [https://arxiv.org/html/2409.12191](https://arxiv.org/html/2409.12191)  
9. Qwen/Qwen2-VL-7B-Instruct \- Hugging Face, 7월 23, 2025에 액세스, [https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)  
10. Unlocking the Future of AI with Qwen 2.5 VL: Where Vision Meets Language \- Alibaba Cloud, 7월 23, 2025에 액세스, [https://www.alibabacloud.com/blog/unlocking-the-future-of-ai-with-qwen-2-5-vl-where-vision-meets-language\_602123](https://www.alibabacloud.com/blog/unlocking-the-future-of-ai-with-qwen-2-5-vl-where-vision-meets-language_602123)  
11. Alibaba Cloud Releases Latest AI Models For Enhanced Visual Understanding and Long Context Inputs, 7월 23, 2025에 액세스, [https://www.alibabacloud.com/blog/alibaba-cloud-releases-latest-ai-models-for-enhanced-visual-understanding-and-long-context-inputs\_601963](https://www.alibabacloud.com/blog/alibaba-cloud-releases-latest-ai-models-for-enhanced-visual-understanding-and-long-context-inputs_601963)  
12. Qwen2. 5-VL Technical Report, 7월 23, 2025에 액세스, [https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)  
13. Qwen2.5 VL\! Qwen2.5 VL\! Qwen2.5 VL\! | Qwen, 7월 23, 2025에 액세스, [https://qwenlm.github.io/blog/qwen2.5-vl/](https://qwenlm.github.io/blog/qwen2.5-vl/)  
14. Qwen \- Wikipedia, 7월 23, 2025에 액세스, [https://en.wikipedia.org/wiki/Qwen](https://en.wikipedia.org/wiki/Qwen)  
15. Qwen2.5-VL \- a Qwen Collection \- Hugging Face, 7월 23, 2025에 액세스, [https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5)  
16. qwen2.5vl \- Ollama, 7월 23, 2025에 액세스, [https://ollama.com/library/qwen2.5vl](https://ollama.com/library/qwen2.5vl)  
17. Qwen2.5-VL-32B: Smarter and Lighter \- Simon Willison's Weblog, 7월 23, 2025에 액세스, [https://simonwillison.net/2025/Mar/24/qwen25-vl-32b/](https://simonwillison.net/2025/Mar/24/qwen25-vl-32b/)  
18. Qwen2.5-VL-32B: A Leaner, Smarter Multimodal Model for Visual Reasoning \- Wandb, 7월 23, 2025에 액세스, [https://wandb.ai/byyoung3/ml-news/reports/Qwen2-5-VL-32B-A-Leaner-Smarter-Multimodal-Model-for-Visual-Reasoning---VmlldzoxMTk2Nzk0NA](https://wandb.ai/byyoung3/ml-news/reports/Qwen2-5-VL-32B-A-Leaner-Smarter-Multimodal-Model-for-Visual-Reasoning---VmlldzoxMTk2Nzk0NA)  
19. QwenLM/Qwen2.5-Omni \- GitHub, 7월 23, 2025에 액세스, [https://github.com/QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)  
20. Qwen/Qwen2.5-VL-3B/7B/72B-Instruct are out\!\! : r/LocalLLaMA \- Reddit, 7월 23, 2025에 액세스, [https://www.reddit.com/r/LocalLLaMA/comments/1itq30t/qwenqwen25vl3b7b72binstruct\_are\_out/](https://www.reddit.com/r/LocalLLaMA/comments/1itq30t/qwenqwen25vl3b7b72binstruct_are_out/)  
21. Alibaba Cloud Model Studio, 7월 23, 2025에 액세스, [https://www.alibabacloud.com/help/en/model-studio/models](https://www.alibabacloud.com/help/en/model-studio/models)  
22. Building Multimodal Services with Qwen and Model Studio \- Alibaba Cloud Community, 7월 23, 2025에 액세스, [https://www.alibabacloud.com/blog/building-multimodal-services-with-qwen-and-model-studio\_600962](https://www.alibabacloud.com/blog/building-multimodal-services-with-qwen-and-model-studio_600962)  
23. Introducing Qwen-VL, 7월 23, 2025에 액세스, [https://qwenlm.github.io/blog/qwen-vl/](https://qwenlm.github.io/blog/qwen-vl/)  
24. Alibaba's Open-Source AI Journey: Innovation, Collaboration, and Future Visions, 7월 23, 2025에 액세스, [https://www.alibabacloud.com/blog/alibabas-open-source-ai-journey-innovation-collaboration-and-future-visions\_602026](https://www.alibabacloud.com/blog/alibabas-open-source-ai-journey-innovation-collaboration-and-future-visions_602026)  
25. How to Install Qwen 2.5 VL Locally? \- NodeShift, 7월 23, 2025에 액세스, [https://nodeshift.com/blog/how-to-install-qwen-2-5-vl-locally](https://nodeshift.com/blog/how-to-install-qwen-2-5-vl-locally)  
26. DocVQA test Benchmark (Visual Question Answering (VQA)) \- Papers With Code, 7월 23, 2025에 액세스, [https://paperswithcode.com/sota/visual-question-answering-on-docvqa-test](https://paperswithcode.com/sota/visual-question-answering-on-docvqa-test)  
27. Qwen2-VL | OpenLM.ai, 7월 23, 2025에 액세스, [https://openlm.ai/qwen2-vl/](https://openlm.ai/qwen2-vl/)  
28. Qwen2-VL — A new milestone in Vision-Language AI \- UnfoldAI, 7월 23, 2025에 액세스, [https://unfoldai.com/qwen2-vl/](https://unfoldai.com/qwen2-vl/)  
29. prithivMLmods/Qwen2-VL-Math-Prase-2B-Instruct \- Hugging Face, 7월 23, 2025에 액세스, [https://huggingface.co/prithivMLmods/Qwen2-VL-Math-Prase-2B-Instruct](https://huggingface.co/prithivMLmods/Qwen2-VL-Math-Prase-2B-Instruct)  
30. MathVista: Evaluating Math Reasoning in Visual Contexts, 7월 23, 2025에 액세스, [https://mathvista.github.io/](https://mathvista.github.io/)  
31. Qwen2.5-VL Vision Model: Features, Applications, and More \- Analytics Vidhya, 7월 23, 2025에 액세스, [https://www.analyticsvidhya.com/blog/2025/01/qwen2-5-vl-vision-model/](https://www.analyticsvidhya.com/blog/2025/01/qwen2-5-vl-vision-model/)  
32. GPU System Requirements Guide for Qwen LLM Models (All Variants), 7월 23, 2025에 액세스, [https://apxml.com/posts/gpu-system-requirements-qwen-models](https://apxml.com/posts/gpu-system-requirements-qwen-models)  
33. Qwen Hosting: Deploy Qwen 1B–72B (VL/AWQ/Instruct) Models Efficiently \- Database Mart, 7월 23, 2025에 액세스, [https://www.databasemart.com/llm/qwen](https://www.databasemart.com/llm/qwen)  
34. Qwen-2.5 Minimum System Requirements: Hardware & Software Specs for Local Installation, 7월 23, 2025에 액세스, [https://www.oneclickitsolution.com/centerofexcellence/aiml/qwen-2-5-minimum-requirements-hardware-software](https://www.oneclickitsolution.com/centerofexcellence/aiml/qwen-2-5-minimum-requirements-hardware-software)  
35. Fine-Tuning a Vision Language Model (Qwen2-VL-7B) | by Amit Yadav \- Medium, 7월 23, 2025에 액세스, [https://medium.com/@amit25173/fine-tuning-a-vision-language-model-qwen2-vl-7b-45e78be66d30](https://medium.com/@amit25173/fine-tuning-a-vision-language-model-qwen2-vl-7b-45e78be66d30)  
36. How to Install Qwen2.5-VL-7B-Instruct Locally \- NodeShift, 7월 23, 2025에 액세스, [https://nodeshift.com/blog/how-to-install-qwen2-5-vl-7b-instruct-locally](https://nodeshift.com/blog/how-to-install-qwen2-5-vl-7b-instruct-locally)  
37. Complete Guide to Fine-tuning Qwen2.5 VL Model \- F22 Labs, 7월 23, 2025에 액세스, [https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen2-5-vl-model/](https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen2-5-vl-model/)  
38. vLLM \- Qwen, 7월 23, 2025에 액세스, [https://qwen.readthedocs.io/en/stable/deployment/vllm.html](https://qwen.readthedocs.io/en/stable/deployment/vllm.html)  
39. vLLM \- Qwen docs, 7월 23, 2025에 액세스, [https://qwen.readthedocs.io/en/latest/deployment/vllm.html](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)  
40. Fine-Tuning | Quantize | Infer — Qwen2-VL mLLM on Custom Data for OCR: Part 3 \- Medium, 7월 23, 2025에 액세스, [https://bhavyajoshi809.medium.com/fine-tuning-qwen2-vl-mllm-on-custom-data-for-ocr-part-3-quantization-of-custom-qwen2-vl-2b-mllm-2c94577f83a5](https://bhavyajoshi809.medium.com/fine-tuning-qwen2-vl-mllm-on-custom-data-for-ocr-part-3-quantization-of-custom-qwen2-vl-2b-mllm-2c94577f83a5)  
41. AWQ \- Qwen docs, 7월 23, 2025에 액세스, [https://qwen.readthedocs.io/en/latest/quantization/awq.html](https://qwen.readthedocs.io/en/latest/quantization/awq.html)  
42. A detailed comparison between GPTQ, AWQ, EXL2, q4\_K\_M, q4\_K\_S, and load\_in\_4bit: perplexity, VRAM, speed, model size, and loading time. : r/LocalLLaMA \- Reddit, 7월 23, 2025에 액세스, [https://www.reddit.com/r/LocalLLaMA/comments/17fuqr7/a\_detailed\_comparison\_between\_gptq\_awq\_exl2\_q4\_k/](https://www.reddit.com/r/LocalLLaMA/comments/17fuqr7/a_detailed_comparison_between_gptq_awq_exl2_q4_k/)  
43. GWQ: Gradient-Aware Weight Quantization for Large Language Models \- arXiv, 7월 23, 2025에 액세스, [https://arxiv.org/html/2411.00850v1](https://arxiv.org/html/2411.00850v1)  
44. Qwen2.5 \- more parameters or less quantization? : r/LocalLLaMA \- Reddit, 7월 23, 2025에 액세스, [https://www.reddit.com/r/LocalLLaMA/comments/1gnomrv/qwen25\_more\_parameters\_or\_less\_quantization/](https://www.reddit.com/r/LocalLLaMA/comments/1gnomrv/qwen25_more_parameters_or_less_quantization/)  
45. Precision Issues with GPTQ-Quantized Qwen2.5-VL Model \#1629 \- GitHub, 7월 23, 2025에 액세스, [https://github.com/vllm-project/llm-compressor/issues/1629](https://github.com/vllm-project/llm-compressor/issues/1629)  
46. Alibaba Cloud Model Studio:Billing for model services, 7월 23, 2025에 액세스, [https://www.alibabacloud.com/help/en/model-studio/billing-for-model-studio](https://www.alibabacloud.com/help/en/model-studio/billing-for-model-studio)  
47. Aliyun Cuts Prices Again: Qwen-VL Large Model Fully Reduced, Process 600 Images for 1 Yuan \- AIbase, 7월 23, 2025에 액세스, [https://www.aibase.com/news/14392](https://www.aibase.com/news/14392)  
48. Chinese tech giants continue to slash prices of LLMs to power AI chatbots \- Global Times, 7월 23, 2025에 액세스, [https://www.globaltimes.cn/page/202501/1326098.shtml](https://www.globaltimes.cn/page/202501/1326098.shtml)  
49. When Self-Hosting AI Models Makes Financial Sense | by Thomasnahon \- Medium, 7월 23, 2025에 액세스, [https://medium.com/@thomasnahon/when-self-hosting-ai-models-makes-financial-sense-3d7cbe11b22c](https://medium.com/@thomasnahon/when-self-hosting-ai-models-makes-financial-sense-3d7cbe11b22c)  
50. Hosted API vs Self-Hosted: Best Deployment Option \- Thinkfree.com, 7월 23, 2025에 액세스, [https://thinkfree.com/hosted-api-vs-self-hosted/](https://thinkfree.com/hosted-api-vs-self-hosted/)  
51. LLM API's vs. Self-Hosting Models : r/LocalLLM \- Reddit, 7월 23, 2025에 액세스, [https://www.reddit.com/r/LocalLLM/comments/1kxlcja/llm\_apis\_vs\_selfhosting\_models/](https://www.reddit.com/r/LocalLLM/comments/1kxlcja/llm_apis_vs_selfhosting_models/)  
52. Qwen/Qwen2-VL-7B-Instruct · Finetuning script using HuggingFace (No llama-factory), 7월 23, 2025에 액세스, [https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/discussions/32](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/discussions/32)  
53. \[Bug\]: Qwen VL 2.5 doesn't work in v0.8.0 \- again \#15122 \- GitHub, 7월 23, 2025에 액세스, [https://github.com/vllm-project/vllm/issues/15122](https://github.com/vllm-project/vllm/issues/15122)  
54. Problem with Qwen2.5-VL-7b \- General \- vLLM Forums, 7월 23, 2025에 액세스, [https://discuss.vllm.ai/t/problem-with-qwen2-5-vl-7b/1065](https://discuss.vllm.ai/t/problem-with-qwen2-5-vl-7b/1065)  
55. Understanding Qwen-v1: My Personal Take | by tangbasky | Data Science Collective, 7월 23, 2025에 액세스, [https://medium.com/data-science-collective/understanding-qwen-v1-my-personal-take-f7b302111d31](https://medium.com/data-science-collective/understanding-qwen-v1-my-personal-take-f7b302111d31)  
56. Exploring Biases in GPT-4o, Claude, and Qwen2.5 Judgements | Simon P. Couch, 7월 23, 2025에 액세스, [https://www.simonpcouch.com/blog/2025-01-30-llm-biases/](https://www.simonpcouch.com/blog/2025-01-30-llm-biases/)  
57. (PDF) Ethical Considerations and Bias Mitigation in Large Language Models AUTHOR, 7월 23, 2025에 액세스, [https://www.researchgate.net/publication/387295166\_Ethical\_Considerations\_and\_Bias\_Mitigation\_in\_Large\_Language\_Models\_AUTHOR](https://www.researchgate.net/publication/387295166_Ethical_Considerations_and_Bias_Mitigation_in_Large_Language_Models_AUTHOR)