# Day 14 Final Integration Evidence Collector Hotfix V2

## 확인된 원인

Collector V1은 Classification Checkpoint 후보를 다음처럼 잘못 예상했다.

```text
models/day4_resnet18_best.pt
models/classification/day4_resnet18_best.pt
models/resnet18_best.pt
```

실제 Day 4 학습 Script와 README가 사용하는 경로는 다음이다.

```text
models/checkpoints/resnet18_transfer_best.pt
```

따라서 `classification_checkpoint : False`는 실제 파일 누락이 아니라
Collector 후보 경로의 오탐이었다.

## 변경 내용

1. 실제 Classification Checkpoint 경로 추가
2. Classification Checkpoint를 최종 Evidence 필수 경로로 지정
3. 실제 경로 탐지 테스트 추가
4. 파일 누락 시 WARN 검증 테스트 추가
5. Schema Version을 2로 변경

## 예상 결과

```text
classification_checkpoint : True |
models/checkpoints/resnet18_transfer_best.pt

Overall status : PASS
Warnings       : 0
```
