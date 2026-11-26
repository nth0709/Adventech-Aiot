# Adventech AIoT: 5축 로봇팔 레이저 선 추적

팔 끝에 카메라와 레이저를 함께 달았습니다(eye-in-hand). 레이저 점이 종이 위 검정 선을
**파란 점(출발)에서 빨간 점(도착)까지** 따라갑니다. Jetson에서 실행합니다.

## 과제와 결과

| 과제 | 기록 | 결과 |
|---|---|---|
| 직선 추적 | `laser_tracker/runs/follow_run7.*` | 파란 점에서 빨간 점까지 따라감. 단, 레이저 픽셀을 고정값으로 쓰던 버전이라 실제 레이저가 선에서 약 30px 벗어나 있었음 |
| Z자 추적 | `laser_tracker/runs/z_run11.*` | 9.8초 주행, 20Hz, 중간 정지 0회. 레이저-선 거리 중앙값 5.8px / 90% 16px / 최대 23px(꼭짓점) |

각 기록은 녹화 영상(`.avi`)과 프레임별 로그(`.csv`)로 되어 있습니다.

## 동작 방식 (`laser_tracker/follow5.py`)
- **폐루프**: 매 프레임 지금 보이는 것만 씁니다. 레이저 점, 검정 선, 파란 점과 빨간 점이 대상입니다. 예측하거나 기억해 둔 위치로는 움직이지 않습니다. 쓸 수 있는 게 안 보이면 멈추고, 1초가 지나면 중단합니다.
- **캘리브레이션**: 관측 자세 주변의 격자 자세(베이스 7 × 상하 5)에서 이미지 자코비안을 한 번 측정해 `calib_grid.json`에 저장하고, 이후 실행은 이 값을 그대로 씁니다. 관측 자세나 레이저 픽셀이 바뀌면 다시 측정하며, `--recalibrate`로 직접 다시 측정할 수도 있습니다.
- **경로**: 현재 프레임에서 선을 따라 거리를 잽니다(geodesic). 그래서 Z자의 예각 꼭짓점도 따라갑니다. 진행 방향은 최근 진행 방향으로 정하고, 꼭짓점에서는 속도를 40%로 줄입니다.

## 실행
```bash
cd laser_tracker
python3 follow5.py --save-rest   # 쉬는 자세 저장 (토크 off, 읽기만)
python3 follow5.py --teach       # 카메라가 종이를 보는 관측 자세 저장
./run.sh run.out --record out.avi   # 추적 (서보 포트가 안 열리면 자동 재시도)
```
- `track5.py`는 `~/so101_arm/servo_bus.py`를 불러옵니다. 저장소의 `so101_arm/` 폴더를 홈 디렉터리에 두거나(`~/so101_arm`), `track5.py`의 `sys.path` 줄을 고쳐 주세요.
- 필요 패키지: `opencv-python`, `numpy`, `pyserial`
- `laser_tracker/README.md`는 초기 3축(Arduino) 버전 문서입니다.

## 구성
| 경로 | 역할 |
|---|---|
| `laser_tracker/follow5.py` | 5축 폐루프 선 추적 (메인) |
| `laser_tracker/track5.py`, `path5.py` | 서보 제어와 자코비안, 선 위 경로 계산 |
| `laser_tracker/laser_spot.py`, `line_detect.py`, `marks.py`, `sheet.py`, `vision.py` | 레이저 점, 선, 파란·빨간 점, 종이 영역 검출 |
| `laser_tracker/run.sh` | 실행 래퍼 (CH343 포트가 멈추면 재시도) |
| `laser_tracker/*.json` | 캘리브레이션과 저장된 자세 |
| `laser_tracker/firmware/` | 초기 3축 버전용 Arduino 펌웨어 |
| `so101_arm/` | STS3215 서보 버스 드라이버, 점검 스크립트 |
