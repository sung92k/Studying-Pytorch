# 변경 로그
MONAI에 대한 모든 주목할만한 변경 사항은 이 파일에 문서화되어 있습니다.

형식은 [Keep Changelog](http://keepachangelog.com/en/1.0.0/)를 기반으로 합니다.
그리고 이 프로젝트는 [Semantic Versioning](http://semver.org/spec/v2.0.0.html)을 따릅니다.

## [미공개]

## [0.7.0] - 2021-09-24
### 추가됨
* [v0.7의 새로운 기능] 개요(docs/source/whatsnew_0_7.md)
* PyTorch 및 NumPy의 입력 및 백엔드를 지원하기 위해 `monai.transforms'의 주요 사용성 개선의 초기 단계
* 일반적인 사용 사례를 위한 [프로파일링 및 튜닝 가이드](https://github.com/Project-MONAI/tutorials/blob/master/acceleration/fast_model_training_guide.md)를 통한 성능 향상
* 최첨단 Kaggle 대회 솔루션의 [교육 모듈 및 워크플로](https://github.com/Project-MONAI/tutorials/tree/master/kaggle/RANZCR/4th_place_solution) 재현
* 다음을 포함한 24개의 새로운 변형
  * `OneOf` 메타 변환
  * 대화형 분할을 위한 DeepEdit 안내 신호 변환
  * 자체 지도 사전 교육을 위한 변환
  * [NVIDIA 도구 확장](https://developer.nvidia.com/blog/nvidia-tools-extension-api-nvtx-annotation-tool-for-profiling-code-in-python-and-cc/) (NVTX) 통합
  * [cuCIM](https://github.com/rapidsai/cucim) 통합
  * 디지털 병리학에 대한 얼룩 정규화 및 상황별 그리드
* 흉부 X선 분석을 위한 시각 언어 변환기를 위한 'Transchex' 네트워크
* `monai.data`의 `DatasetSummary` 유틸리티
* `워밍업코사인스케줄`
* 더 나은 이전 버전과의 호환성을 위한 사용 중단 경고 및 문서 지원
* 추가 `kwargs` 및 다른 백엔드 API로 패딩
* 다양한 네트워크 및 해당 하위 모듈의 'dropout' 및 'norm'과 같은 추가 옵션

### 변경됨
* 기본 Docker 이미지가 `nvcr.io/nvidia/pytorch:21.06-py3`에서 `nvcr.io/nvidia/pytorch:21.08-py3`으로 업그레이드됨
* 더 이상 사용되지 않는 입력 인수 `n_classes`, 대신 `num_classes`
* 더 이상 사용되지 않는 입력 인수 `dimensions` 및 `ndims`, `spatial_dims` 사용
* 더 나은 가독성을 위해 Sphinx 기반 문서 테마 업데이트
* `NdarrayTensor` 유형은 더 간단한 주석을 위해 `NdarrayOrTensor`로 대체되었습니다.
* 자기 주의 기반 네트워크 블록은 이제 2D 및 3D 입력을 모두 지원합니다.

### 제거됨
* 더 이상 사용되지 않는 `TransformInverter`, `monai.transforms.InvertD` 사용
* 야간 및 병합 후 테스트를 위한 GitHub 자체 호스팅 CI/CD 파이프라인
* `monai.handlers.utils.evenly_divisible_all_gather`
* `monai.handlers.utils.string_list_all_gather`

### 수정됨
* `LMDBDataset`의 다중 스레드 캐시 쓰기 문제
* 이미지 판독기의 출력 모양 규칙 불일치
* `NiftiSaver`, `PNGSaver`의 출력 디렉토리 및 파일 이름 유연성 문제
* test-time Augmentation에서 'label' 필드의 요구사항
* `ThreadDataLoader`에 대한 입력 인수 유연성 문제
* 분리된 `Dice`와 `CrossEntropy` 중간 결과가 `DiceCELoss`
* 다양한 모듈의 문서, 코드 예제 및 경고 메시지 개선
* 사용자가 보고한 다양한 사용성 문제

## [0.6.0] - 2021-07-08
### 추가됨
* 10개의 새로운 변환, 마스크된 손실 래퍼 및 전이 학습을 위한 'NetAdapter'
* Clara Train [Medical Model ARchives(MMAR)](https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html)에서 네트워크 및 사전 훈련된 가중치를 로드하는 API
* 기본 메트릭 및 누적 메트릭 API, 4개의 새로운 회귀 메트릭
* 초기 CSV 데이터 세트 지원
* 기본 첫 번째 후처리 단계로 미니 배치 데콜레이트, [v0.5 코드를 v0.6으로 마이그레이션](https://github.com/Project-MONAI/MONAI/wiki/v0.5-to-v0.6-migration-guide) wiki는 주요 변경 사항에 적응하는 방법을 보여줍니다.
* `monai.utils.deprecated`를 통한 초기 역호환성 지원
* 주의 기반 비전 모듈 및 세분화를 위한 'UNETR'
* PyTorch JIT 컴파일을 사용하는 일반 모듈 로더 및 가우스 혼합 모델
* 이미지 패치 샘플링 변환의 역
* 네트워크 차단 유틸리티 `get_[norm, act, dropout, pool]_layer`
* `apply_transform` 및 `Compose`에 대한 `unpack_items` 모드
* Deepgrow 대화형 워크플로의 새로운 이벤트 'INNER_ITERATION_STARTED'
* 캐시 기반 데이터세트를 위한 `set_data` API가 데이터세트 콘텐츠를 동적으로 업데이트합니다.
* PyTorch 1.9와 완벽하게 호환
* `runtests.sh`에 대한 `--disttests` 및 `--min` 옵션
* Nvidia Blossom 시스템과의 사전 병합 테스트 초기 지원

### 변경됨
* 기본 Docker 이미지가 다음에서 `nvcr.io/nvidia/pytorch:21.06-py3`으로 업그레이드되었습니다.
  `nvcr.io/nvidia/pytorch:21.04-py3`
* 선택적으로 v0.4.4 대신 PyTorch-Ignite v0.4.5에 의존
* 데모, 튜토리얼, 테스트 데이터를 프로젝트, [`Project-MONAI/MONAI-extra-test-data`](https://github.com/Project-MONAI/MONAI-extra-test-data)를 공유 드라이브에 통합
* 용어 통합: `post_transform`이 `postprocessing`으로, `pre_transform`이 `preprocessing`으로 이름이 변경되었습니다.
* "채널 우선" 데이터 형식을 허용하도록 후처리 변환 및 이벤트 핸들러를 통합했습니다.
* `evenly_divisible_all_gather` 및 `string_list_all_gather`가 `monai.utils.dist`로 이동됨

### 제거됨
* 후처리 변환 및 이벤트 핸들러를 위한 '일괄 처리' 입력 지원
* `TorchVisionFullyConvModel`
* `set_visible_devices` 유틸리티 함수
* `SegmentationSaver` 및 `TransformsInverter` 핸들러

### 수정됨
* 빅엔디안 이미지 헤더 처리 문제
* 캐시 기반 데이터세트에서 비무작위 변환에 대한 다중 스레드 문제
* 존재하지 않는 캐시 위치를 여러 프로세스가 공유할 때 데이터 세트 문제가 지속됨
* Numpy 1.21.0의 타이핑 문제
* `strict_shape=False`일 때 `CheckpointLoader`를 사용하여 `model`과 `optmizier`로 체크포인트 로드
* `SplitChannel`은 numpy/torch 입력에 따라 동작이 다릅니다.
* Lambda 함수로 인한 변환 피클링 문제
* `generate_param_groups`에서 이름으로 필터링하는 문제
* `class_activation_maps`의 반환 값 유형 불일치
* 다양한 독스트링 오타
* `monai.transforms`의 다양한 사용성 향상

## [0.5.3] - 2021-05-28
### 변경됨
* 프로젝트 기본 분기의 이름이 `master`에서 `dev`로 변경됨
* 기본 Docker 이미지가 `nvcr.io/nvidia/pytorch:21.02-py3`에서 `nvcr.io/nvidia/pytorch:21.04-py3`으로 업그레이드됨
* `iteration_metric` 핸들러에 대한 향상된 유형 검사
* 캐싱 계산 중에 `tempfile`을 사용하도록 향상된 `PersistentDataset`
* 향상된 다양한 정보/오류 메시지
* `RandAffine`의 향상된 성능
* `SmartCacheDataset`의 향상된 성능
* 플랫폼이 `Linux`인 경우 선택적으로 `cucim`이 필요합니다.
* `TestTimeAugmentation`의 기본 `장치`가 `cpu`로 변경됨

### 수정됨
* 이제 다운로드 유틸리티가 더 나은 기본 매개변수를 제공합니다.
* 패치 기반 변환에서 중복된 `key_transforms`
* `ClassificationSaver`의 다중 GPU 문제
* `SpacingD`의 기본 `meta_data` 문제
* 영구 데이터 로더 작업자의 데이터 세트 캐싱 문제
* `permutohedral_cuda`의 메모리 문제
* `CopyItemsd`의 사전 키 문제
* deepgrow `SpatialCropForegroundd`에 대한 `box_start` 및 `box_end` 매개변수
* `MaskedInferenceWSIDataset`의 조직 마스크 배열 전치 문제
* 다양한 유형 힌트 오류
* 다양한 독스트링 오타

### 추가됨
* `TransformInverter`에 대한 `to_tensor` 및 `device` 인수 지원
* SpatialCrop을 사용한 슬라이싱 옵션
* 이전 버전과의 호환성을 위한 네트워크의 클래스 이름 별칭
* CropForeground에 대한 `k_divisible` 옵션
* `Compose`에 대한 `map_items` 옵션
* 표면 거리 계산을 위한 'inf' 및 'nan' 경고
* 이미지 보호기에 대한 `print_log` 플래그
* Python 3.9용 기본 테스트 파이프라인

## [0.5.0] - 2021-04-09
### 추가됨
* 개요 문서 [v0.5.0의 주요 기능](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* 뒤집을 수 있는 공간 변환
  * `InvertibleTransform` 기본 API
  * 일괄 역방향 및 디콜레이트 API
  * '작성'의 역
  * 일괄 역 이벤트 처리
  * 응용 프로그램으로 테스트 시간 증강
* 학습 기반 이미지 등록 초기 지원:
  * 굽힘 에너지, LNCC 및 글로벌 상호 정보 손실
  * 완전 컨볼루션 아키텍처
  * 조밀한 변위장, 조밀한 속도장 계산
  * C++/CUDA 구현을 통한 고차 보간으로 워핑
* 대화형 세분화를 위한 Deepgrow 모듈:
  * 클릭 시뮬레이션 워크플로
  * 안내 신호를 위한 거리 기반 변환
* 디지털 병리학 지원:
  * Nvidia cuCIM 및 SmartCache를 사용한 효율적인 전체 슬라이드 이미징 IO 및 샘플링
  * 병변에 대한 FROC 측정
  * 병변 검출을 위한 확률적 후처리
  * 완전한 컨볼루션 분석을 위한 TorchVision 분류 모델 어댑터
* 12개의 새로운 변환, 그리드 패치 데이터 세트, `ThreadDataLoader`, EfficientNets B0-B7
* 워크플로를 보다 세밀하게 제어하기 위한 엔진의 4가지 반복 이벤트
* 새로운 C++/CUDA 확장:
  * 조건부 랜덤 필드
  * Permutohedral 격자를 사용한 빠른 양방향 필터링
* 메트릭 요약 보고 및 API 저장
* DiceCELoss, DiceFocalLoss, 분할 손실 계산을 위한 다중 스케일 래퍼
* 데이터 로딩 유틸리티:
  * `decollate_batch`
  * 역 지원이 있는 `PadListDataCollate`
* `Dataset`에 대한 슬라이싱 구문 지원
* 손실 모듈에 대한 초기 Torchscript 지원
* 학습률 찾기
* 사전 기반 변환에서 누락된 키 허용
* 전이 학습을 위한 체크포인트 로딩 지원
* Jupyter 노트북을 위한 다양한 요약 및 플로팅 유틸리티
* 기여자 서약 행동 강령
* 튜토리얼 리포지토리를 다루는 주요 CI/CD 개선 사항
* PyTorch 1.8과 완벽하게 호환
* Nvidia Blossom 인프라를 사용한 초기 야간 CI/CD 파이프라인

### 변경됨
* 향상된 `list_data_collate` 오류 처리
* 통합 반복 메트릭 API
* `densenet*` 확장 프로그램의 이름이 `DenseNet*`으로 변경됨
* `se_res*` 네트워크 확장의 이름이 `SERes*`로 변경됨
* 변환 기반 API가 'compose', 'inverse', 'transform'으로 재정렬됨
* 무작위 증강을 위한 `_do_transform` 플래그는 `RandomizableTransform`을 통해 통합됩니다.
* 분리된 후처리 단계, 예: 메트릭 계산의 `softmax`, `to_onehot_y`
* 배포된 샘플러를 `monai.data.utils`에서 `monai.data.samplers`로 이동했습니다.
* 엔진의 데이터 로더는 이제 일반 이터러블을 입력으로 받아들입니다.
* 워크플로는 이제 추가 사용자 지정 이벤트 및 상태 속성을 허용합니다.
* Numpy 1.20에 따른 다양한 타입 힌트
* `--unittest` 및 `--net`(통합 테스트) 옵션을 갖도록 테스트 유틸리티 `runtests.sh`를 리팩토링했습니다.
* 기본 Docker 이미지가 `nvcr.io/nvidia/pytorch:20.10-py3`에서 `nvcr.io/nvidia/pytorch:21.02-py3`으로 업그레이드됨
* Docker 이미지는 이제 자체 호스팅 환경으로 빌드됩니다.
* 기본 연락처 이메일이 `monai.contact@gmail.com`으로 업데이트되었습니다.
* 이제 GitHub 토론을 기본 커뮤니케이션 포럼으로 사용합니다.

### 제거됨
* PyTorch 1.5.x에 대한 호환성 테스트
* 특정 로더 형식 지정, 예: `LoadNifti`, `NiftiDataset`
* 테스트가 아닌 파일에서 진술을 주장
* `from module import *` 문, 해결 flake8 F403

### 수정됨
* PyTorch에 따라 코드에 미국 영어 철자를 사용합니다.
* 코드 적용 범위는 이제 다중 처리 실행을 고려합니다.
* 초기 셔플링이 있는 SmartCache
* `ConvertToMultiChannelBasedOnBratsClasses`는 이제 채널 우선 입력을 지원합니다.
* 루트가 아닌 권한으로 저장하기 위한 체크포인트 핸들러
* 분산 단위 테스트 종료 문제 수정
* 깊은 감독 없이 단일 텐서 출력을 갖도록 통합된 `DynUNet`
* `SegmentationSaver`는 이제 사용자 지정 데이터 유형과 `squeeze_end_dims` 플래그를 지원합니다.
* `*Saver` 이벤트 핸들러가 `data_root_dir` 옵션을 사용하여 파일 이름을 출력하는 문제를 수정했습니다.
* 이제 이미지 로드 기능이 리틀 엔디안을 보장합니다.
* 정규식 기반 테스트 케이스 일치를 지원하도록 테스트 실행기를 수정했습니다.
* 이벤트 핸들러의 사용성 문제

## [0.4.0] - 2020-12-15
### 추가됨
* 개요 문서 [v0.4.0의 주요 기능](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* 네트 모듈에 대한 Torchscript 지원
* 새로운 네트워크 및 레이어:
  * 이산 가우스 커널
  * 힐베르트 변환 및 엔벨로프 감지
  * Swish 및 mish 활성화
  * Acti-norm-dropout 블록
  * 업샘플링 레이어
  * Autoencoder, Variational autoencoder
  * FC넷
* densitynet, senet, multichannel AHNet에 대해 사전 훈련된 가중치에서 초기화 지원
* 레이어별 학습률 API
* 폐색 감도, 혼동 매트릭스, 표면 거리를 기반으로 하는 새로운 모델 메트릭 및 이벤트 핸들러
* CAM/GradCAM/GradCAM++
* Nibabel, ITK 리더를 사용하는 파일 형식에 구애받지 않는 이미지 로더 API
* 데이터세트 파티션, 교차 검증 API 개선
* 새로운 데이터 API:
  * LMDB 기반 캐싱 데이터 세트
  * Cache-N-transforms 데이터 세트
  * 반복 가능한 데이터 세트
  * 패치 데이터 세트
* 주간 PyPI 릴리스
* PyTorch 1.7과 완벽하게 호환
* CI/CD 개선 사항:
  * 건너 뛰기, 속도 향상, 빨리 실패, 시간 지정, 빠른 테스트
  * 분산 교육 테스트
  * 성능 프로파일링 유틸리티
* 새로운 튜토리얼 및 데모:
  * 오토인코더, VAE 튜토리얼
  * 교차 검증 데모
  * 모델 해석 가능성 튜토리얼
  * COVID-19 폐 CT 세분화 도전 오픈 소스 기준선
  * 스레드 버퍼 데모
  * 데이터세트 파티셔닝 튜토리얼
  * 레이어별 학습률 데모
  * [MONAI 부트캠프 2020](https://github.com/Project-MONAI/MONAIBootcamp2020)

### 변경됨
* 기본 Docker 이미지가 `nvcr.io/nvidia/pytorch:20.08-py3`에서 `nvcr.io/nvidia/pytorch:20.10-py3`으로 업그레이드됨

#### 이전 버전과 호환되지 않는 변경 사항
* `monai.apps.CVDecathlonDataset`은 `dataset_cls` 옵션이 있는 일반 `monai.apps.CrossValidation`으로 확장됩니다.
* 이제 캐시 데이터 세트에는 변환 인수로 `monai.transforms.Compose` 인스턴스가 필요합니다.
* 모델 체크포인트 파일 이름 확장자가 `.pth`에서 `.pt`로 변경됨
* 독자의 `get_spatial_shape`는 목록 대신 numpy 배열을 반환합니다.
* 메트릭 및 이벤트 핸들러에서 `sigmoid`, `to_onehot_y`, `mutually_exclusive`, `logit_thresh`와 같은 후처리 단계를 분리했습니다.
메트릭 메서드를 호출하기 전에 후처리 단계를 사용해야 합니다.
* `ConfusionMatrixMetric` 및 `DiceMetric` 계산은 이제 유효한 결과를 나타내기 위해 추가 `not_nans` 플래그를 반환합니다.
* `UpSample` 옵션인 `mode`는 이제 `"deconv"`, `"nontrainable"`, `"pixelshuffle"`을 지원합니다. `interp_mode`는 `mode`가 `"nontrainable"`인 경우에만 사용됩니다.
* `SegResNet` 옵션인 `upsample_mode`는 이제 `"deconv"`, `"nontrainable"`, `"pixelshuffle"`을 지원합니다.
* `monai.transforms.Compose` 클래스는 `monai.transforms.Transform`을 상속합니다.
* `Rotate`, `Rotated`, `RandRotate`, `RandRotated` 변환에서 `angle` 관련 매개변수는 각도가 아닌 라디안 단위의 각도로 해석됩니다.
* `SplitChannel` 및 `SplitChannel`이 `transforms.post`에서 `transforms.utility`로 이동됨

### 제거됨
* PyTorch 1.4 지원

### 수정됨
* 안정성과 유연성을 위한 향상된 손실 기능
* 슬라이딩 윈도우 추론 메모리 및 장치 문제
* 수정된 변환:
  * 강도 데이터 유형 및 정규화 유형 정규화
  * 확대/축소를 위한 패딩 모드
  * 자르기 좌표 반환
  * 선택 항목 변형
  * 가중치 패치 샘플링
  * 확대/축소를 위한 종횡비를 유지하는 옵션
* 다양한 CI/CD 이슈

## [0.3.0] - 2020-10-02
### 추가됨
* 개요 문서 [v0.3.0의 기능 주요 기능](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* 자동 혼합 정밀도 지원
* 다중 노드, 다중 GPU 데이터 병렬 모델 교육 지원
* 3개의 새로운 평가 메트릭 기능
* 11개의 새로운 네트워크 레이어 및 블록
* 6개의 새로운 네트워크 아키텍처
* I/O 어댑터를 포함한 14개의 새로운 변환
* `DecathlonDataset`에 대한 교차 검증 모듈
* 데이터세트의 스마트 캐시 모듈
* `monai.optimizers` 모듈
* `monai.csrc` 모듈
* ITK, Nibabel, Numpy, Pillow(PIL Fork)를 사용한 ImageReader의 실험적 기능
* C++/CUDA에서 미분 가능한 이미지 리샘플링의 실험적 기능
* 앙상블 평가 모듈
* GAN 트레이너 모듈
* C++/CUDA 코드용 초기 크로스 플랫폼 CI 환경
* 코드 스타일 적용에는 이제 isort 및 clang 형식이 포함됩니다.
* tqdm이 있는 진행률 표시줄

### 변경됨
* 이제 PyTorch 1.6과 완벽하게 호환됩니다.
* 기본 Docker 이미지가 `nvcr.io/nvidia/pytorch:20.03-py3`에서 `nvcr.io/nvidia/pytorch:20.08-py3`으로 업그레이드되었습니다.
* 이제 코드 기여는 [개발자 원산지 증명서(DCO)](https://developercertificate.org/)에 서명해야 합니다.
* Type 힌팅 주요 작업 완료
* [Open Data on AWS](https://registry.opendata.aws/)로 마이그레이션된 원격 데이터 세트
* 선택적으로 v0.3.0 대신 PyTorch-Ignite v0.4.2에 의존
* 선택적으로 토치비전, ITK에 의존
* 8개의 새로운 테스트 환경으로 향상된 CI 테스트

### 제거됨
* `MONAI/examples` 폴더([`Project-MONAI/tutorials`](https://github.com/Project-MONAI/tutorials)로 재배치됨)
* `MONAI/research` 폴더([`Project-MONAI/research-contributions`](https://github.com/Project-MONAI/research-contributions)로 재배치됨)

### 수정됨
* `dense_patch_slices` 잘못된 인덱싱
* 'GeneralizedWassersteinDiceLoss'의 데이터 유형 문제
* `ZipDataset` 반환 값 불일치
* `sliding_window_inference` 인덱싱 및 `device` 문제
* monai 모듈을 가져오면 네임스페이스 오염이 발생할 수 있습니다.
* `DecathlonDataset`에서 무작위 데이터 분할 문제
* 'Compose' 변환 무작위화 문제
* 함수 유형 힌트의 다양한 문제
* 독스트링 및 문서의 오타
* 기존 파일 폴더의 'PersistentDataset' 문제
* 출력 작성자의 파일 이름 문제

## [0.2.0] - 2020-07-02
### 추가됨
* 개요 문서 [v0.2.0의 기능 주요 기능](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* 유형 힌트 및 정적 유형 분석 지원
* `MONAI/research` 폴더
* 지도 교육을 위한 `monai.engine.workflow` API
* 유효성 검사 및 추론을 위한 `monai.inferers` API
* 7개의 새로운 튜토리얼 및 예제
* 3개의 새로운 손실 기능
* 4개의 새로운 이벤트 핸들러
* 8개의 새로운 레이어, 블록 및 네트워크
* 후처리 변환을 포함한 12개의 새로운 변환
* `MedNISTDataset` 및 `DecathlonDataset`을 포함한 `monai.apps.datasets` API
* 영구 캐싱, `ZipDataset` 및 `monai.data`의 `ArrayDataset`
* 여러 Python 버전을 지원하는 플랫폼 간 CI 테스트
* 선택적 가져오기 메커니즘
* 타사 변환 통합을 위한 실험적 기능

### 변경됨
> 자세한 내용은 [프로젝트 위키](https://github.com/Project-MONAI/MONAI/wiki/Notable-changes-between-0.1.0-and-0.2.0)를 참조하세요.
* 이제 핵심 모듈에는 numpy >= 1.17이 필요합니다.
* `monai.transforms` 모듈을 자르기 및 패드, 강도, IO, 후처리, 공간 및 유틸리티로 분류했습니다.
* 이제 대부분의 변환이 PyTorch 기본 API로 구현됩니다.
* 코드 스타일 적용 및 자동화된 형식 지정 워크플로는 이제 autopep8 및 black을 사용합니다.
* 기본 Docker 이미지가 `nvcr.io/nvidia/pytorch:19.10-py3`에서 `nvcr.io/nvidia/pytorch:20.03-py3`으로 업그레이드됨
* 향상된 로컬 테스트 도구
* 문서 웹사이트 도메인이 https://docs.monai.io 로 변경되었습니다.

### 제거됨
* Python < 3.6 지원
* pytorch-ignite, nibabel, tensorboard, pillow, scipy, scikit-image를 포함한 선택적 종속성의 자동 설치

### 수정됨
* 유형 및 인수 이름 일관성의 다양한 문제
* docstring 및 문서 사이트의 다양한 문제
* 단위 및 통합 테스트의 다양한 문제
* 예제 및 노트북의 다양한 문제

## [0.1.0] - 2020-04-17
### 추가됨
* Apache 2.0 라이선스에 따른 공개 알파 소스 코드 릴리스([주요 기능](https://github.com/Project-MONAI/MONAI/blob/0.1.0/docs/source/highlights.md))
* 다양한 튜토리얼 및 예제
  - 의료 이미지 분류 및 세분화 워크플로
  - CPU/GPU 및 캐싱을 통한 간격/방향 인식 전처리
  - PyTorch Ignite 및 Lightning을 사용한 유연한 워크플로
* 다양한 GitHub 작업
  - 자체 호스팅 러너를 통한 CI/CD 파이프라인
  - readthedocs.org를 통한 문서 출판
  - PyPI 패키지 퍼블리싱
* 기여 지침
* 프로젝트 로고 및 배지

[주요 기능]: https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md

[미공개]: https://github.com/Project-MONAI/MONAI/compare/0.7.0...HEAD
[0.7.0]: https://github.com/Project-MONAI/MONAI/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/Project-MONAI/MONAI/compare/0.5.3...0.6.0
[0.5.3]: https://github.com/Project-MONAI/MONAI/compare/0.5.0...0.5.3
[0.5.0]: https://github.com/Project-MONAI/MONAI/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/Project-MONAI/MONAI/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/Project-MONAI/MONAI/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/Project-MONAI/MONAI/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/Project-MONAI/MONAI/commits/0.1.0
