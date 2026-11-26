/*
 * Laser line-tracker firmware for Arduino Mega 2560
 *  - Jetson <-> Mega : USB Serial (Serial, 115200), ASCII line commands
 *  - Mega  <-> STS3215 x3 : Serial1 (TX1=18, RX1=19), 1 Mbps, Feetech STS protocol
 *  - Laser : digital pin LASER_PIN (via transistor/MOSFET module)
 *
 * The STS bus is single-wire half-duplex. Use a half-duplex adapter
 * (e.g. Feetech FE-URT-1 / Waveshare bus servo adapter) or a TX-resistor/buffer
 * circuit. If TX and RX are tied together the Mega receives its own bytes (echo);
 * the packet reader below detects and skips that echo automatically.
 *
 * Host commands (one per line, '\n' terminated):
 *   PING            -> "PONG <id>:<0|1> ..."
 *   R               -> "POS p1 p2 p3"  (-1 = read failure)
 *   P p1 p2 p3      -> sync-write goal positions (0..4095), no reply
 *   V speed acc     -> set goal speed (0=max, steps/s) and acceleration (0..254) -> "OK"
 *   T 0|1           -> torque off/on for all servos -> "OK"
 *   L 0|1           -> laser off/on -> "OK"
 *   H               -> heartbeat (keeps watchdog alive), no reply
 * Any command resets the watchdog. If nothing arrives for WATCHDOG_MS the laser
 * is switched off (servos keep holding their last position).
 */

#define HOST Serial
#define BUS  Serial1

// ---------------- user configuration ----------------
const long     HOST_BAUD   = 115200;
const long     BUS_BAUD    = 1000000;   // STS3215 factory default
const uint8_t  NUM_SERVOS  = 3;
const uint8_t  SERVO_IDS[NUM_SERVOS] = {1, 2, 3};  // base(yaw), shoulder, elbow
const int      LASER_PIN   = 8;
const bool     LASER_ACTIVE_HIGH = true;
const unsigned long WATCHDOG_MS = 1000;
const unsigned long BUS_TIMEOUT_US = 3000;
// ----------------------------------------------------

// STS instructions / registers
const uint8_t INST_PING = 0x01, INST_READ = 0x02, INST_WRITE = 0x03, INST_SYNC_WRITE = 0x83;
const uint8_t REG_TORQUE_ENABLE = 40, REG_ACC = 41, REG_PRESENT_POS = 56;
const uint8_t BROADCAST_ID = 0xFE;

uint16_t goalSpeed = 0;   // 0 = maximum speed
uint8_t  goalAcc   = 0;   // 0 = maximum acceleration
unsigned long lastCmdMs = 0;
bool laserOn = false;

uint8_t txBuf[64];
uint8_t txLen = 0;

// ---------------- STS low-level ----------------
void busSend(uint8_t id, uint8_t inst, const uint8_t *params, uint8_t n) {
  uint8_t len = n + 2;
  uint8_t sum = id + len + inst;
  txLen = 0;
  txBuf[txLen++] = 0xFF; txBuf[txLen++] = 0xFF;
  txBuf[txLen++] = id;   txBuf[txLen++] = len; txBuf[txLen++] = inst;
  for (uint8_t i = 0; i < n; i++) { txBuf[txLen++] = params[i]; sum += params[i]; }
  txBuf[txLen++] = ~sum;
  while (BUS.available()) BUS.read();     // drop stale bytes
  BUS.write(txBuf, txLen);
  BUS.flush();
}

int busReadByte() {
  unsigned long t0 = micros();
  while (!BUS.available()) {
    if (micros() - t0 > BUS_TIMEOUT_US) return -1;
  }
  return BUS.read();
}

// Read one status packet from `id`. Returns payload length (excluding error byte) or -1.
int busReceive(uint8_t id, uint8_t *data, uint8_t maxN) {
  for (uint8_t attempt = 0; attempt < 2; attempt++) {   // attempt 0 may be our own echo
    // find header 0xFF 0xFF
    int prev = -1, c;
    for (uint8_t k = 0; k < 16; k++) {
      c = busReadByte();
      if (c < 0) return -1;
      if (prev == 0xFF && c == 0xFF) break;
      prev = c;
    }
    if (!(prev == 0xFF && c == 0xFF)) return -1;
    int rid = busReadByte(); if (rid < 0) return -1;
    if (rid == 0xFF) { rid = busReadByte(); if (rid < 0) return -1; }  // 3rd 0xFF
    int len = busReadByte(); if (len < 2 || len > 40) return -1;
    uint8_t pkt[42];
    uint8_t sum = rid + len;
    for (int i = 0; i < len; i++) {
      int b = busReadByte(); if (b < 0) return -1;
      pkt[i] = (uint8_t)b;
      if (i < len - 1) sum += pkt[i];
    }
    if ((uint8_t)(~sum) != pkt[len - 1]) return -1;
    // echo detection: identical to what we just transmitted
    bool echo = (txLen == (uint8_t)(len + 4)) && (txBuf[2] == rid) && (txBuf[3] == len);
    for (int i = 0; echo && i < len; i++) if (txBuf[4 + i] != pkt[i]) echo = false;
    if (echo) continue;
    if (rid != id) return -1;
    uint8_t n = len - 2;                 // pkt[0]=error, pkt[1..n]=data
    if (n > maxN) n = maxN;
    for (uint8_t i = 0; i < n; i++) data[i] = pkt[1 + i];
    return n;
  }
  return -1;
}

bool stsPing(uint8_t id) {
  busSend(id, INST_PING, NULL, 0);
  uint8_t d[4];
  return busReceive(id, d, sizeof(d)) >= 0;
}

int stsReadPos(uint8_t id) {
  uint8_t p[2] = {REG_PRESENT_POS, 2};
  busSend(id, INST_READ, p, 2);
  uint8_t d[2];
  if (busReceive(id, d, 2) != 2) return -1;
  int v = d[0] | (d[1] << 8);
  if (v & 0x8000) v = -(v & 0x7FFF);     // STS sign bit
  return v;
}

void stsWriteByte(uint8_t id, uint8_t reg, uint8_t val) {
  uint8_t p[2] = {reg, val};
  busSend(id, INST_WRITE, p, 2);
  if (id != BROADCAST_ID) { uint8_t d[2]; busReceive(id, d, 2); }  // consume status
}

// Sync write: ACC(41), GOAL_POS(42-43), GOAL_TIME(44-45), GOAL_SPEED(46-47)
void stsSyncWritePos(const int *pos) {
  const uint8_t DLEN = 7;
  uint8_t params[2 + NUM_SERVOS * (1 + DLEN)];
  uint8_t k = 0;
  params[k++] = REG_ACC;
  params[k++] = DLEN;
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    int v = constrain(pos[i], 0, 4095);
    params[k++] = SERVO_IDS[i];
    params[k++] = goalAcc;
    params[k++] = v & 0xFF;          params[k++] = (v >> 8) & 0xFF;
    params[k++] = 0;                 params[k++] = 0;            // goal time
    params[k++] = goalSpeed & 0xFF;  params[k++] = (goalSpeed >> 8) & 0xFF;
  }
  busSend(BROADCAST_ID, INST_SYNC_WRITE, params, k);   // broadcast: no status reply
}

// ---------------- laser ----------------
void setLaser(bool on) {
  laserOn = on;
  digitalWrite(LASER_PIN, (on == LASER_ACTIVE_HIGH) ? HIGH : LOW);
}

// ---------------- host command handling ----------------
char line[64];
uint8_t lineLen = 0;

void handleCommand(char *s) {
  char *tok = strtok(s, " \t\r");
  if (!tok) return;
  lastCmdMs = millis();

  if (!strcmp(tok, "P")) {
    int pos[NUM_SERVOS];
    for (uint8_t i = 0; i < NUM_SERVOS; i++) {
      char *a = strtok(NULL, " \t\r");
      if (!a) { HOST.println(F("ERR args")); return; }
      pos[i] = atoi(a);
    }
    stsSyncWritePos(pos);
  } else if (!strcmp(tok, "H")) {
    // heartbeat only
  } else if (!strcmp(tok, "R")) {
    HOST.print(F("POS"));
    for (uint8_t i = 0; i < NUM_SERVOS; i++) { HOST.print(' '); HOST.print(stsReadPos(SERVO_IDS[i])); }
    HOST.println();
  } else if (!strcmp(tok, "PING")) {
    HOST.print(F("PONG"));
    for (uint8_t i = 0; i < NUM_SERVOS; i++) {
      HOST.print(' '); HOST.print(SERVO_IDS[i]); HOST.print(':'); HOST.print(stsPing(SERVO_IDS[i]) ? 1 : 0);
    }
    HOST.println();
  } else if (!strcmp(tok, "V")) {
    char *a = strtok(NULL, " \t\r"), *b = strtok(NULL, " \t\r");
    if (!a || !b) { HOST.println(F("ERR args")); return; }
    goalSpeed = (uint16_t)atol(a);
    goalAcc = (uint8_t)constrain(atoi(b), 0, 254);
    HOST.println(F("OK"));
  } else if (!strcmp(tok, "T")) {
    char *a = strtok(NULL, " \t\r");
    uint8_t on = (a && atoi(a)) ? 1 : 0;
    for (uint8_t i = 0; i < NUM_SERVOS; i++) stsWriteByte(SERVO_IDS[i], REG_TORQUE_ENABLE, on);
    HOST.println(F("OK"));
  } else if (!strcmp(tok, "L")) {
    char *a = strtok(NULL, " \t\r");
    setLaser(a && atoi(a));
    HOST.println(F("OK"));
  } else {
    HOST.println(F("ERR cmd"));
  }
}

void setup() {
  pinMode(LASER_PIN, OUTPUT);
  setLaser(false);
  HOST.begin(HOST_BAUD);
  BUS.begin(BUS_BAUD);
  lastCmdMs = millis();
  HOST.println(F("READY"));
}

void loop() {
  while (HOST.available()) {
    char c = HOST.read();
    if (c == '\n') {
      line[lineLen] = 0;
      handleCommand(line);
      lineLen = 0;
    } else if (lineLen < sizeof(line) - 1) {
      line[lineLen++] = c;
    }
  }
  if (laserOn && millis() - lastCmdMs > WATCHDOG_MS) {
    setLaser(false);
    HOST.println(F("WDT laser off"));
  }
}
