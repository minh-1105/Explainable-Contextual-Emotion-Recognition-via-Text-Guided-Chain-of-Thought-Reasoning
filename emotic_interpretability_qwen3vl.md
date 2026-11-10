# Interpretability Module for Explainable Contextual Emotion Recognition

## 1. Mục tiêu

Sau khi mô hình Emotion Recognition được train và thực hiện inference trên ảnh EMOTIC, mục tiêu của Interpretability Module là trả lời câu hỏi:

> **Tại sao mô hình lại dự đoán emotion này?**

Qwen3-VL không trực tiếp đọc được các feature nội bộ của mô hình emotion. Vì vậy, cần một bước trung gian để trích xuất evidence từ chính mô hình đã dự đoán emotion, sau đó mới dùng Qwen3-VL để chuyển evidence đó thành lời giải thích bằng ngôn ngữ tự nhiên.

Pipeline tổng quát:

```text
EMOTIC Image
     │
     ▼
Emotion Recognition Model
     │
     ├── Emotion Prediction
     │      ├── Happiness = 0.82
     │      └── Engagement = 0.71
     │
     ▼
Interpretability Module
     │
     ├── Feature Attribution
     ├── Visual Token Attribution
     ├── Attention Analysis
     ├── Integrated Gradients / Grad-CAM
     ├── Occlusion
     └── Important Regions / Relations
             │
             ▼
        Evidence Package
             │
             ▼
          Qwen3-VL
             │
             ▼
Grounded Natural-Language Explanation
```

---

# 2. Emotion Recognition Model

Mô hình emotion là model chính thực hiện dự đoán.

Ví dụ kiến trúc:

```text
Person Image ──► Person Encoder ──┐
                                  │
Background ────► Context Encoder ─┼──► Fusion ─► Emotion Head
                                  │
Scene Graph ───► Graph Encoder ───┘
```

Output:

```text
Affection   = 0.21
Happiness   = 0.82
Engagement  = 0.71
Pleasure    = 0.54
Sadness     = 0.08
...
```

Với EMOTIC, output có thể là 26 emotion categories và VAD nếu cần.

---

# 3. Interpretability Module

Interpretability Module có nhiệm vụ xác định:

- feature nào quan trọng,
- visual token nào quan trọng,
- vùng ảnh nào ảnh hưởng prediction,
- relation nào trong scene graph ảnh hưởng prediction,
- prediction thay đổi thế nào khi loại bỏ evidence.

Không nên chỉ hỏi Qwen:

```text
Image + Happiness = 0.82
→ "Explain why"
```

vì Qwen có thể tạo ra một explanation hợp lý về mặt ngôn ngữ nhưng không phản ánh đúng cơ chế ra quyết định của model emotion.

Thay vào đó:

```text
Emotion Model
     ↓
Interpretability
     ↓
Model-grounded Evidence
     ↓
Qwen3-VL
     ↓
Explanation
```

---

# 4. Feature Attribution

Feature attribution đánh giá mức độ ảnh hưởng của các feature nội bộ đến một emotion cụ thể.

Ví dụ:

```text
F_person
F_context
F_graph
```

Nếu:

```text
Happiness = 0.82
```

có thể đo:

```text
Person features   = 0.48
Context features  = 0.21
Graph features    = 0.31
```

Hoặc ở mức feature dimension:

```text
person_feature_34 = 0.12
person_feature_81 = 0.09
graph_feature_12  = 0.15
```

Một cách đơn giản là sử dụng gradient:

```math
Importance_i =
\left|
\frac{\partial y_{emotion}}
{\partial F_i}
\right|
```

Ví dụ:

```math
Importance_i =
\left|
\frac{\partial y_{happy}}
{\partial F_i}
\right|
```

Feature có gradient lớn hơn thường ảnh hưởng mạnh hơn đến prediction.

Tuy nhiên, raw feature như:

```text
feature_34
feature_81
```

không có ý nghĩa trực tiếp với con người.

Vì vậy cần ánh xạ chúng về:

- visual regions,
- visual tokens,
- objects,
- relationships,
- semantic concepts.

---

# 5. Visual Token Attribution

Nếu encoder dùng Vision Transformer (ViT), ảnh được chia thành patches:

```text
Image
 ↓
Patches
 ↓
[v1][v2][v3]...[vN]
```

Mỗi visual token đại diện cho một vùng ảnh.

Ta đo:

```math
Importance(v_i, y)
```

với `y` là emotion cần giải thích.

Ví dụ:

```text
Token 41 → upper body      = 0.26
Token 42 → left arm        = 0.18
Token 87 → nearby person   = 0.24
Token 91 → table           = 0.05
Token 120 → wall           = 0.01
```

Sau đó map visual tokens về vị trí ảnh để tạo heatmap.

Ví dụ:

```text
Most important regions:

1. Upper body
2. Nearby person
3. Arms
4. Face

Low importance:

- Wall
- Generic background
```

Visual token attribution giúp trả lời:

> **Model đã dùng vùng nào của ảnh để tạo emotion prediction?**

---

# 6. Attention Analysis

Attention cho biết một token trong Transformer đang tham chiếu mạnh đến các token nào khác.

Ví dụ emotion representation cho `Happiness`:

```text
nearby_person token   = 0.30
target_body token     = 0.27
face token            = 0.16
restaurant token      = 0.08
```

Nếu có scene graph:

```text
target_person ──facing──► person_2
```

relation token này cũng có thể nhận attention cao.

Ví dụ:

```text
Happiness representation attends to:

target → facing → person_2   = 0.23
target upper body            = 0.18
target arms                  = 0.14
nearby person                = 0.13
target → near → person_2     = 0.11
```

## Lưu ý

Attention không đồng nghĩa với causal explanation.

Không nên kết luận:

> Attention cao nghĩa là đây chắc chắn là nguyên nhân.

Attention nên được sử dụng như:

- visualization,
- auxiliary analysis,
- evidence bổ sung.

Cần kết hợp với intervention như occlusion.

---

# 7. Integrated Gradients / Grad-CAM

## 7.1 Grad-CAM

Grad-CAM tạo heatmap cho biết vùng ảnh nào có ảnh hưởng lớn đến output.

Ví dụ:

```text
Happiness prediction
        ↓
Grad-CAM
        ↓
Heatmap
```

Heatmap có thể highlight:

```text
Face
Upper body
Arms
Nearby person
```

và ít highlight:

```text
Wall
Floor
Generic background
```

Grad-CAM rất phù hợp cho visualization trong paper.

---

## 7.2 Integrated Gradients

Integrated Gradients đo contribution của input từ một baseline đến input thật.

Công thức:

```math
IG_i =
(x_i-x'_i)
\int_0^1
\frac{\partial F(x'+\alpha(x-x'))}
{\partial x_i}
d\alpha
```

Kết quả có thể được tổng hợp thành:

```text
Integrated Gradients:

Target upper body = 0.31
Nearby person     = 0.27
Face              = 0.17
Background        = 0.05
```

Integrated Gradients phù hợp với:

- Vision Transformer,
- multimodal transformer,
- token-based architectures.

---

# 8. Occlusion

Occlusion là một trong những phần quan trọng nhất vì nó thực hiện intervention trực tiếp lên input.

Ý tưởng:

> Loại bỏ một vùng hoặc một relation và xem prediction thay đổi bao nhiêu.

Ví dụ:

```text
Original:
Happiness = 0.82
```

Che nearby person:

```text
Happiness = 0.51
```

Che target face:

```text
Happiness = 0.73
```

Che background:

```text
Happiness = 0.79
```

Tính:

```math
\Delta_{nearby}
=
0.82 - 0.51
=
0.31
```

```math
\Delta_{face}
=
0.82 - 0.73
=
0.09
```

```math
\Delta_{background}
=
0.82 - 0.79
=
0.03
```

Suy ra:

```text
Nearby person >> Face >> Generic background
```

Điều này cho thấy social context ảnh hưởng mạnh hơn background chung.

---

# 9. Important Regions

Sau khi chạy attribution, các token/feature cần được map thành các region có ý nghĩa semantic.

Ví dụ:

```json
{
  "important_regions": [
    {
      "region": "target_upper_body",
      "score": 0.34
    },
    {
      "region": "nearby_person",
      "score": 0.29
    },
    {
      "region": "face",
      "score": 0.18
    }
  ]
}
```

Region có thể lấy từ:

- object detector,
- person bounding box,
- segmentation,
- ViT patches,
- scene graph nodes.

Mục tiêu của bước này là chuyển:

```text
token_41
token_87
feature_34
```

thành:

```text
upper body
nearby person
face
```

để Qwen3-VL có thể giải thích.

---

# 10. Scene Graph Relation Attribution

Nếu hệ thống có Scene Graph, mỗi relation có thể được xem như một reasoning unit.

Ví dụ graph:

```text
Target Person
   │
   ├── facing ─────────► Person 2
   ├── sitting_near ───► Person 2
   ├── holding ────────► Glass
   └── located_in ─────► Restaurant
```

Các relation có thể được biểu diễn thành relation tokens:

```text
r1 = [target, facing, person_2]
r2 = [target, sitting_near, person_2]
r3 = [target, holding, glass]
r4 = [target, located_in, restaurant]
```

Ta đo contribution của từng relation:

```math
Importance(r_i)
=
\left|
\frac{\partial y_{happy}}
{\partial r_i}
\right|
```

Ví dụ:

```text
facing(person_2)          = 0.31
sitting_near(person_2)    = 0.27
holding(glass)            = 0.08
located_in(restaurant)    = 0.11
```

Như vậy có thể nói:

```text
Most important relationships:

1. target facing another person
2. target sitting near another person
3. restaurant context
```

---

# 11. Causal Validation cho Scene Graph

Không chỉ nhìn attribution score.

Có thể remove relation và chạy inference lại.

Ví dụ:

```text
Original:

target → facing → person_2

Happiness = 0.82
```

Remove relation:

```text
Happiness = 0.59
```

Ta có:

```math
\Delta = 0.82 - 0.59 = 0.23
```

Điều này cung cấp evidence mạnh hơn rằng relation có ảnh hưởng đến prediction.

---

# 12. Evidence Package

Sau khi Interpretability Module hoàn thành, toàn bộ evidence được chuẩn hóa thành một cấu trúc thống nhất.

Ví dụ:

```json
{
  "prediction": {
    "happiness": 0.82,
    "engagement": 0.71
  },

  "feature_contribution": {
    "person": 0.48,
    "context": 0.21,
    "graph": 0.31
  },

  "important_regions": [
    {
      "region": "target_upper_body",
      "importance": 0.34
    },
    {
      "region": "nearby_person",
      "importance": 0.29
    },
    {
      "region": "face",
      "importance": 0.18
    }
  ],

  "important_relations": [
    {
      "relation": "target_person facing person_2",
      "importance": 0.31
    },
    {
      "relation": "target_person sitting_near person_2",
      "importance": 0.27
    }
  ],

  "occlusion_test": [
    {
      "removed": "nearby_person",
      "happiness_before": 0.82,
      "happiness_after": 0.51
    },
    {
      "removed": "face",
      "happiness_before": 0.82,
      "happiness_after": 0.73
    }
  ],

  "relation_ablation": [
    {
      "removed_relation": "target_person facing person_2",
      "happiness_before": 0.82,
      "happiness_after": 0.59
    }
  ]
}
```

Evidence Package là đầu vào chính cho Explanation Generator.

---

# 13. Vai trò của Qwen3-VL

Qwen3-VL không phải model quyết định emotion.

Vai trò của Qwen:

```text
Model-grounded Evidence
        ↓
Qwen3-VL
        ↓
Human-readable Explanation
```

Nó hoạt động như một:

> **Explanation Generator**

hoặc:

> **Evidence-to-Language Translator**

Qwen nhận:

1. Original image
2. Target person bounding box
3. Emotion prediction
4. Important regions
5. Attribution scores
6. Occlusion results
7. Important graph relations
8. Relation ablation results

---

# 14. Prompt cho Qwen3-VL

Ví dụ:

```text
You are given the prediction of an emotion recognition model
and model-derived interpretability evidence.

Explain the prediction using only the supplied evidence.

Do not introduce visual cues or reasons that are not supported
by the attribution or intervention results.

Emotion prediction:
- Happiness: 0.82
- Engagement: 0.71

Important regions:
- target upper body: 0.34
- nearby person: 0.29
- face: 0.18

Important relations:
- target person facing person_2: 0.31
- target person sitting near person_2: 0.27

Occlusion:
- removing nearby person changes Happiness from 0.82 to 0.51
- removing face changes Happiness from 0.82 to 0.73

Relation ablation:
- removing target-person-facing-person_2 changes Happiness
  from 0.82 to 0.59

Generate a concise explanation grounded strictly in this evidence.
```

---

# 15. Ví dụ output từ Qwen3-VL

```text
The model's Happiness prediction is primarily influenced by
the target person's body posture and interaction with a nearby
individual.

The nearby person is particularly important because removing
this region reduces the Happiness score from 0.82 to 0.51.

The relation between the target person and the nearby individual
also contributes strongly. Removing the "facing" relationship
reduces the Happiness score from 0.82 to 0.59.

In contrast, removing the face produces a smaller decrease.
This suggests that the prediction is driven more strongly by
body posture and interpersonal context than by facial information alone.
```

Điểm quan trọng:

> Mỗi statement trong explanation phải trace được về evidence.

---

# 16. Không nên làm

Không nên sử dụng:

```text
Image
+
Emotion prediction
+
"Why?"
```

Ví dụ:

```text
Image
Happiness = 0.82

Prompt:
Why is this person happy?
```

Qwen có thể hallucinate:

```text
The person is happy because they are smiling at a party.
```

trong khi Emotion Model có thể thực tế dựa chủ yếu vào:

```text
body posture
+
nearby person
```

Đây là:

> **Post-hoc hallucinated explanation**

và không phản ánh đúng decision process của model.

---

# 17. Thiết kế Interpretability Module đề xuất

Không cần sử dụng tất cả các phương pháp với cùng trọng số.

Có thể gom thành ba tầng:

```text
Interpretability Module
│
├── 1. Attribution
│   ├── Integrated Gradients
│   ├── Visual Token Attribution
│   └── Attention Analysis
│
├── 2. Causal Validation
│   ├── Region Occlusion
│   └── Relation Ablation
│
└── 3. Semantic Mapping
    ├── Important Regions
    ├── Objects
    └── Scene Graph Relations
```

Sau đó:

```text
Interpretability Module
        ↓
Evidence Package
        ↓
Qwen3-VL
        ↓
Grounded Natural-Language Explanation
```

---

# 18. Bộ phương pháp khuyến nghị

Đối với EMOTIC + Scene Graph + Explainable Emotion Recognition, nên ưu tiên:

## Primary methods

```text
Integrated Gradients
+
Region Occlusion
+
Scene Graph Relation Attribution
+
Relation Ablation
```

## Supporting method

```text
Attention Visualization
```

Attention nên được dùng để:

- visualization,
- supplementary analysis,
- layer/token inspection,

không nên dùng một mình để chứng minh causal importance.

---

# 19. Pipeline hoàn chỉnh đề xuất

```text
                          EMOTIC
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
           Person Crop              Full Image
                │                       │
                ▼                       ▼
          Person Encoder          Context Encoder
                │                       │
          Visual Tokens            Context Tokens
                │                       │
                └──────────┬────────────┘
                           │
                     Scene Graph
                           │
                    Relation Tokens
                           │
                           ▼
                 Multimodal Fusion
                           │
                           ▼
                    Emotion Head
                           │
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
         Happiness     Engagement        ...
           0.82           0.71
             │
             ▼
       Interpretability Module
             │
       ┌─────┼──────────────┐
       ▼     ▼              ▼
 Attribution  Occlusion   Relation Analysis
       │       │              │
       └───────┼──────────────┘
               ▼
        Evidence Package
               │
               ▼
           Qwen3-VL
               │
               ▼
 Grounded Natural-Language Explanation
```

---

# 20. Research Contribution

Hệ thống có thể được mô tả như:

> **An interpretable contextual emotion recognition framework that explains predictions through visual-token, region-level, and relation-level attribution, while using a vision-language model to translate model-grounded evidence into human-readable explanations.**

Contribution chính không phải là:

> Qwen giải thích emotion.

Mà là:

```text
Emotion Prediction
        +
Model-grounded Attribution
        +
Causal Validation
        +
Scene Graph Reasoning
        +
Evidence-grounded Language Explanation
```

---

# 21. Tóm tắt

Vai trò của từng thành phần:

| Thành phần | Vai trò |
|---|---|
| Emotion Model | Dự đoán emotion |
| Feature Attribution | Xác định feature nội bộ quan trọng |
| Visual Token Attribution | Xác định patch/token ảnh quan trọng |
| Attention | Phân tích token interaction |
| Integrated Gradients | Attribution cho input/token |
| Grad-CAM | Visualization vùng ảnh |
| Occlusion | Kiểm chứng bằng intervention |
| Scene Graph Attribution | Xác định relation quan trọng |
| Relation Ablation | Kiểm chứng causal importance của relation |
| Evidence Package | Chuẩn hóa toàn bộ evidence |
| Qwen3-VL | Chuyển evidence thành explanation |
| Natural-language Explanation | Output cuối cho người dùng |

Tóm gọn:

```text
Model predicts emotion
        ↓
Interpretability finds evidence
        ↓
Intervention validates evidence
        ↓
Evidence Package
        ↓
Qwen3-VL verbalizes evidence
        ↓
Grounded Explanation
```

Qwen3-VL không được xem là nguồn gốc của explanation evidence.

Evidence phải đến từ chính mô hình emotion và các phương pháp interpretability / intervention.
