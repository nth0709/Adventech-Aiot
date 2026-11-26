# 레이저 검정선 추적 (3축 STS3215 팔, eye-in-hand)

```
Jetson ──USB(115200)──> Arduino Mega 2560 ──Serial1 1Mbps(반이중)──> STS3215 ID 1,2,3
  │                         └─ D8 ──> 레이저 모듈(트랜지스터/MOSFET 경유)
  └─USB3──> Arducam B0495 (/dev/video0, 팔 끝에 레이저와 함께 장착)
```

## 빠른 실행 (`init.sh`)
```bash
./init.sh setup     # 최초 1회 (sudo): dialout 그룹 + ModemManager가 Arduino를 건드리지 않게 udev 규칙
./init.sh flash     # 펌웨어 컴파일·업로드
./init.sh check     # 장비 점검
./init.sh calib     # 캘리브레이션
./init.sh track     # 추적 (옵션 그대로 전달: --hold --headless --record out.avi)
./init.sh all       # flash → check → calib(calib.json 없을 때) → track
./init.sh           # check → calib(없을 때) → track
```
dialout 그룹이 현재 세션에 아직 반영되지 않았으면 자동으로 `sg dialout`으로 재실행합니다.
포트·Arduino 경로는 환경변수 `PORT`, `ARDUINO_DIR`로 바꿀 수 있습니다.

## 파일
| 파일 | 역할 |
|---|---|
| `firmware/laser_tracker_fw/laser_tracker_fw.ino` | Mega 펌웨어 (STS 프로토콜 직접 구현, 외부 라이브러리 불필요) |
| `config.py` | 포트·관절 한계·비전·제어 파라미터 |
| `check_devices.py` | 카메라/시리얼/서보/레이저 점검 |
| `calibrate.py` | home 자세 티칭, 레이저 픽셀 검출, 이미지 자코비안 측정 → `calib.json` |
| `track.py` | 추적 실행 |
| `arm.py`, `vision.py` | 시리얼 통신, 카메라·선 검출 |

## 1. 사전 준비 (최초 1회)
```bash
sudo usermod -aG dialout $USER     # /dev/ttyACM0 권한 → 로그아웃/로그인(또는 재부팅)
```
서보 ID를 1(베이스), 2(어깨), 3(팔꿈치)로 미리 설정하세요(Feetech FD 소프트웨어 등). 다르면 펌웨어 `SERVO_IDS` 수정.

## 2. 펌웨어 업로드
```bash
arduino --upload firmware/laser_tracker_fw/laser_tracker_fw.ino \
        --board arduino:avr:mega:cpu=atmega2560 --port /dev/ttyACM0
```
(또는 Arduino IDE에서 열어 업로드.) 레이저 핀/극성은 `LASER_PIN`, `LASER_ACTIVE_HIGH`에서 변경합니다.

## 3. 점검 → 캘리브레이션 → 추적
```bash
python3 check_devices.py           # PONG 1:1 2:1 3:1, 레이저 3회 깜빡임 확인
python3 calibrate.py               # 토크 OFF → 손으로 작업면을 향하게 → Enter
python3 track.py                   # GUI (q 종료, space 일시정지, f follow/hold, l 레이저, r 방향반전)
python3 track.py --headless        # SSH 환경
```
**캘리브레이션 주의:** 자코비안 측정은 영상 이동량(phase correlation)으로 하므로, 무늬가 있는 평면(인쇄된 종이 등)을 비추세요. 흰 종이에 선 하나만 있으면 선 방향 이동량을 측정할 수 없습니다(`confidence` 낮음 경고). 레이저 픽셀은 흰 면에서 측정하는 것이 좋습니다.

## 4. 튜닝 포인트 (`config.py`)
- **`JOINT_MIN/MAX`**: 가장 먼저 실제 팔의 안전 범위로 설정 (기본값 1024~3072은 임의값)
- 진동 → `GAIN` ↓, `SERVO_ACC` 설정(예: 50) / 느림 → `GAIN` ↑, `MAX_STEP` ↑
- 선 검출 불량 → `DARK_MAX`(조명에 맞게), `MIN_LINE_AREA`, `ROI_MARGIN`
- 따라가는 속도 → `LOOKAHEAD` (0이면 제자리 유지 = `--hold`)

## 동작 원리
1. 선 검출: 회색조 → Otsu ∩ (밝기 < `DARK_MAX`) → 레이저와 가장 가까운 연결요소 선택
2. 조준점: 선 위 최근접점 → PCA로 국소 방향 → `LOOKAHEAD`만큼 앞으로 → 선 단면 중앙으로 보정. 선 끝에 닿으면 반대 방향으로 왕복
3. 제어: 오차 e = 조준점 − 레이저픽셀, dq = GAIN·W Jᵀ(J W Jᵀ+λ²I)⁻¹(−e) + null-space home 복귀항 (3관절 → 2D 오차의 여유 자유도 처리)
4. 안전: 펌웨어 워치독 — 1초간 명령이 없으면 레이저 OFF

## 하드웨어 참고
- STS3215는 단선 반이중 TTL입니다. Mega TX1(18)/RX1(19)를 반이중 어댑터(FE-URT-1, Waveshare 버스서보 어댑터 등) 또는 버퍼 회로로 연결하세요. TX/RX를 직결해 자기 송신 에코가 들어와도 펌웨어가 자동으로 걸러냅니다.
- 서보 전원(6~12V)은 별도 공급하고 GND를 Mega와 공통으로 연결하세요.
