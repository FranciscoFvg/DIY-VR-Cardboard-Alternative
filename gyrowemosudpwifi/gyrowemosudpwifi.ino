#include <ESP8266WiFi.h>      // Uso da placa WEMOS D1 R2
#include <ESP8266WebServer.h> // Servidor web
#include <WiFiUdp.h>          // Envio de pacotes UDP
#include <DFRobot_BMI160.h>   // Comunicação com sensor 6 DOF BMI160
#include <Wire.h>             // Conexão I2C
#include <WiFiManager.h>      // Permite configuração do wifi sem regravação
#include <EEPROM.h>           // Armazenamento de dados na memória do esp8266

DFRobot_BMI160 bmi160;       // inicializando o sensor
WiFiUDP udp;                 // inicializando o protocolo UDP
ESP8266WebServer server(80); // inicializando o servidor web

// ====== BOTÃO FÍSICO ======
const int BUTTON_PIN = D5;  // GPIO14 - Pino do botão
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// ====== OPENTRACK (PC) ======
IPAddress pcIP(192, 168, 0, 14); // IP da máquina com OPENTRACK
unsigned int pcPort = 4242;      // porta usada pelo OPENTRACK

// ====== BMI160 ======
const int8_t i2c_addr = 0x69;

float pitch, roll, yaw = 0;

float pitchAcc = 0, rollAcc = 0;
const float alpha = 0.98; // maior =+ confiança no gyro

float gyroX_offset = 0;
float gyroY_offset = 0;
float gyroZ_offset = 0;

unsigned long lastTime;
float dt;

// ====== EEPROM ======
struct UdpConfig
{
  uint32_t magic;
  uint8_t ip[4];
  uint16_t port;
};
const uint32_t UDP_MAGIC = 0x55445031; // "UDP1"

struct CalibrationData
{
  uint32_t magic;
  float gyroX_offset;
  float gyroY_offset;
  float gyroZ_offset;
};
const uint32_t CALIB_MAGIC = 0x43414C31; // "CAL1"

struct AxisConfig
{
  uint32_t magic;
  bool enableYaw;
  bool enablePitch;
  bool enableRoll;
};
const uint32_t AXIS_MAGIC = 0x41584953; // "AXIS"

// Variáveis de configuração de eixos
bool enableYaw = true;
bool enablePitch = true;
bool enableRoll = true;

void loadUdpConfig()
{
  UdpConfig cfg;
  EEPROM.get(0, cfg);
  if (cfg.magic == UDP_MAGIC)
  {
    pcIP = IPAddress(cfg.ip[0], cfg.ip[1], cfg.ip[2], cfg.ip[3]);
    pcPort = cfg.port;
  }
}

void saveUdpConfig()
{
  UdpConfig cfg;
  cfg.magic = UDP_MAGIC;
  cfg.ip[0] = pcIP[0];
  cfg.ip[1] = pcIP[1];
  cfg.ip[2] = pcIP[2];
  cfg.ip[3] = pcIP[3];
  cfg.port = pcPort;

  EEPROM.put(0, cfg);
  EEPROM.commit();
}

bool loadCalibration()
{
  CalibrationData cal;
  EEPROM.get(sizeof(UdpConfig), cal);
  
  if (cal.magic == CALIB_MAGIC)
  {
    gyroX_offset = cal.gyroX_offset;
    gyroY_offset = cal.gyroY_offset;
    gyroZ_offset = cal.gyroZ_offset;
    
    Serial.println("Calibração carregada da EEPROM:");
    Serial.print("Offset Gyro X = ");
    Serial.println(gyroX_offset);
    Serial.print("Offset Gyro Y = ");
    Serial.println(gyroY_offset);
    Serial.print("Offset Gyro Z = ");
    Serial.println(gyroZ_offset);
    return true;
  }
  return false;
}

void saveCalibration()
{
  CalibrationData cal;
  cal.magic = CALIB_MAGIC;
  cal.gyroX_offset = gyroX_offset;
  cal.gyroY_offset = gyroY_offset;
  cal.gyroZ_offset = gyroZ_offset;
  
  EEPROM.put(sizeof(UdpConfig), cal);
  EEPROM.commit();
  
  Serial.println("Calibração salva na EEPROM!");
}

void loadAxisConfig()
{
  AxisConfig cfg;
  EEPROM.get(sizeof(UdpConfig) + sizeof(CalibrationData), cfg);
  
  if (cfg.magic == AXIS_MAGIC)
  {
    enableYaw = cfg.enableYaw;
    enablePitch = cfg.enablePitch;
    enableRoll = cfg.enableRoll;
    
    Serial.println("Configuração de eixos carregada:");
    Serial.print("Yaw: ");
    Serial.println(enableYaw ? "ON" : "OFF");
    Serial.print("Pitch: ");
    Serial.println(enablePitch ? "ON" : "OFF");
    Serial.print("Roll: ");
    Serial.println(enableRoll ? "ON" : "OFF");
  }
}

void saveAxisConfig()
{
  AxisConfig cfg;
  cfg.magic = AXIS_MAGIC;
  cfg.enableYaw = enableYaw;
  cfg.enablePitch = enablePitch;
  cfg.enableRoll = enableRoll;
  
  EEPROM.put(sizeof(UdpConfig) + sizeof(CalibrationData), cfg);
  EEPROM.commit();
  
  Serial.println("Configuração de eixos salva na EEPROM!");
}

// ====== CALIBRAÇÃO ======
void calibrateGyro()
{
  const int samples = 500;
  long sumX = 0, sumY = 0, sumZ = 0;

  Serial.println("Calibrando Gyro X, Y e Z... NÃO MEXA O SENSOR");
  delay(1000);

  for (int i = 0; i < samples; i++)
  {
    int16_t data[6];
    bmi160.getAccelGyroData(data);
    sumX += data[0]; // Gyro X
    sumY += data[1]; // Gyro Y
    sumZ += data[2]; // Gyro Z
    delay(5);
  }

  gyroX_offset = sumX / (float)samples;
  gyroY_offset = sumY / (float)samples;
  gyroZ_offset = sumZ / (float)samples;

  Serial.print("Offset Gyro X = ");
  Serial.println(gyroX_offset);
  Serial.print("Offset Gyro Y = ");
  Serial.println(gyroY_offset);
  Serial.print("Offset Gyro Z = ");
  Serial.println(gyroZ_offset);

  saveCalibration();
}

// ====== WEB ======
void handleRoot()
{
  String html = "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<title>Head Tracker ESP8266</title>"
                "<style>"
                "body{font-family:Arial,Helvetica,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px;}"
                ".card{max-width:560px;margin:0 auto;background:#111827;border:1px solid #1f2937;"
                "border-radius:16px;padding:20px;box-shadow:0 10px 25px rgba(0,0,0,.35);}"
                "h2{margin:0 0 12px 0;font-size:22px;color:#f8fafc;}"
                ".grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}"
                ".item{background:#0b1220;border:1px solid #1f2937;border-radius:12px;padding:12px;}"
                ".label{font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;}"
                ".value{font-size:20px;font-weight:700;color:#e2e8f0;}"
                ".line{height:1px;background:#1f2937;margin:16px 0;}"
                "input[type=text]{width:100%;padding:10px;border-radius:10px;border:1px solid #334155;"
                "background:#0b1220;color:#e2e8f0;box-sizing:border-box;}"
                ".checkbox-group{display:flex;gap:16px;flex-wrap:wrap;}"
                ".checkbox-item{display:flex;align-items:center;gap:8px;}"
                ".checkbox-item input[type=checkbox]{width:20px;height:20px;cursor:pointer;}"
                ".checkbox-item label{cursor:pointer;color:#e2e8f0;font-size:14px;}"
                "button{padding:10px 14px;border:0;border-radius:10px;background:#22c55e;color:#0f172a;"
                "font-weight:700;cursor:pointer;width:100%;}"
                "button.secondary{background:#38bdf8;}"
                "button.warning{background:#f59e0b;}"
                "form{margin:0;display:flex;flex-direction:column;gap:10px;}"
                ".status-indicator{display:inline-block;width:10px;height:10px;border-radius:50%;margin-left:8px;}"
                ".status-on{background:#22c55e;}"
                ".status-off{background:#ef4444;}"
                "@media (max-width:520px){.grid{grid-template-columns:1fr;}}"
                "</style>"
                "<script>"
                "setInterval(function(){"
                "fetch('/status').then(r=>r.json()).then(d=>{"
                "document.getElementById('pitch').textContent=d.pitch;"
                "document.getElementById('roll').textContent=d.roll;"
                "document.getElementById('yaw').textContent=d.yaw;"
                "document.getElementById('dest').textContent=d.dest;"
                "});"
                "}, 50);"
                "</script>"
                "</head><body>"
                "<div class='card'>"
                "<h2>Head Tracker - ESP8266</h2>"
                "<div class='grid'>"
                "<div class='item'><div class='label'>Pitch</div><div class='value' id='pitch'>" +
                String(pitch, 2) + "<span class='status-indicator " + String(enablePitch ? "status-on" : "status-off") + "'></span></div></div>"
                "<div class='item'><div class='label'>Roll</div><div class='value' id='roll'>" +
                String(roll, 2) + "<span class='status-indicator " + String(enableRoll ? "status-on" : "status-off") + "'></span></div></div>"
                "<div class='item'><div class='label'>Yaw</div><div class='value' id='yaw'>" +
                String(yaw, 2) + "<span class='status-indicator " + String(enableYaw ? "status-on" : "status-off") + "'></span></div></div>"
                "<div class='item'><div class='label'>IP ESP</div><div class='value'>" +
                WiFi.localIP().toString() + "</div></div>"
                "<div class='item' style='grid-column:1/-1;'><div class='label'>Destino UDP</div>"
                "<div class='value' id='dest'>" +
                pcIP.toString() + ":" + String(pcPort) + "</div></div>"
                "</div>"
                "<div class='line'></div>"
                "<form action='/axis' method='get'>"
                "<div class='label' style='margin-bottom:10px;'>Eixos Habilitados</div>"
                "<div class='checkbox-group'>"
                "<div class='checkbox-item'>"
                "<input type='checkbox' id='yaw' name='yaw' value='1'" + String(enableYaw ? " checked" : "") + ">"
                "<label for='yaw'>Yaw</label>"
                "</div>"
                "<div class='checkbox-item'>"
                "<input type='checkbox' id='pitch' name='pitch' value='1'" + String(enablePitch ? " checked" : "") + ">"
                "<label for='pitch'>Pitch</label>"
                "</div>"
                "<div class='checkbox-item'>"
                "<input type='checkbox' id='roll' name='roll' value='1'" + String(enableRoll ? " checked" : "") + ">"
                "<label for='roll'>Roll</label>"
                "</div>"
                "</div>"
                "<button type='submit' style='margin-top:10px;'>Salvar Eixos</button>"
                "</form>"
                "<div class='line'></div>"
                "<form action='/reset'>"
                "<button class='secondary' type='submit'>Zerar Yaw</button>"
                "</form>"
                "<div class='line'></div>"
                "<form action='/calibrate' onsubmit='return confirm(\"Coloque o sensor em superfície plana. Continuar?\");'>"
                "<button class='warning' type='submit'>Recalibrar Giroscópio</button>"
                "</form>"
                "<div class='line'></div>"
                "<form action='/setIP' method='get'>"
                "<input type='text' name='ip' placeholder='IP destino (ex: 192.168.0.14)'>"
                "<input type='text' name='port' placeholder='Porta (ex: 4242)'>"
                "<button type='submit'>Salvar destino</button>"
                "</form>"
                "</div>"
                "</body></html>";

  server.send(200, "text/html", html);
}

void handleReset()
{
  yaw = 0;
  pitch = 0;
  roll = 0;
  Serial.println("Orientação resetada!");
  server.sendHeader("Location", "/");
  server.send(303);
}

void handleCalibrate()
{
  String html = "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<title>Calibrando...</title>"
                "<style>"
                "body{font-family:Arial,Helvetica,sans-serif;background:#0f172a;color:#e2e8f0;"
                "display:flex;align-items:center;justify-content:center;height:100vh;margin:0;}"
                ".card{background:#111827;border:1px solid #1f2937;border-radius:16px;padding:32px;"
                "text-align:center;max-width:400px;}"
                ".spinner{border:4px solid #1f2937;border-top:4px solid #f59e0b;border-radius:50%;"
                "width:48px;height:48px;animation:spin 1s linear infinite;margin:0 auto 20px;}"
                "@keyframes spin{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}"
                "h2{color:#f8fafc;margin:0 0 12px;}"
                "p{color:#94a3b8;margin:0;}"
                "</style>"
                "<meta http-equiv='refresh' content='8;url=/'>"
                "</head><body>"
                "<div class='card'>"
                "<div class='spinner'></div>"
                "<h2>Calibrando Giroscópio</h2>"
                "<p>NÃO MEXA O SENSOR!<br>Redirecionando em 8 segundos...</p>"
                "</div>"
                "</body></html>";
  
  server.send(200, "text/html", html);
  
  delay(500);
  calibrateGyro();
}

void handleStatus()
{
  String json = "{";
  json += "\"pitch\":" + String(pitch, 2) + ",";
  json += "\"roll\":" + String(roll, 2) + ",";
  json += "\"yaw\":" + String(yaw, 2) + ",";
  json += "\"dest\":\"" + pcIP.toString() + ":" + String(pcPort) + "\"";
  json += "}";
  server.send(200, "application/json", json);
}

void handleSetIP()
{
  bool changed = false;

  if (server.hasArg("ip"))
  {
    IPAddress newIP;
    if (newIP.fromString(server.arg("ip")))
    {
      pcIP = newIP;
      changed = true;
    }
  }
  if (server.hasArg("port"))
  {
    int p = server.arg("port").toInt();
    if (p > 0 && p <= 65535)
    {
      pcPort = (unsigned int)p;
      changed = true;
    }
  }

  if (changed)
  {
    saveUdpConfig();
  }

  server.sendHeader("Location", "/");
  server.send(303);
}

void handleAxisConfig()
{
  enableYaw = server.hasArg("yaw");
  enablePitch = server.hasArg("pitch");
  enableRoll = server.hasArg("roll");
  
  saveAxisConfig();
  
  server.sendHeader("Location", "/");
  server.send(303);
}

// ====== SETUP ======
void setup()
{
  Serial.begin(115200);
  delay(2000);

  // Configurar botão com pull-up interno
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  EEPROM.begin(256); // Aumentado para comportar todas as configurações
  loadUdpConfig();   // Carrega configuração UDP salva (se existir)
  loadAxisConfig();  // Carrega configuração de eixos

  // I2C ESP8266
  Wire.begin(D2, D1);

  if (bmi160.I2cInit(i2c_addr) != BMI160_OK)
  {
    Serial.println("Erro ao iniciar BMI160");
    while (1)
      ;
  }

  // Tenta carregar calibração salva, se não existir, calibra
  if (!loadCalibration())
  {
    Serial.println("Calibração não encontrada. Calibrando pela primeira vez...");
    calibrateGyro();
  }

  // ===== WiFiManager =====
  WiFiManager wm;
  wm.setConfigPortalTimeout(180);

  if (!wm.autoConnect("HeadTracker-ESP8266"))
  {
    Serial.println("Falha ao conectar. Reiniciando...");
    ESP.restart();
  }

  Serial.println("\nConectado!");
  Serial.print("IP do ESP: ");
  Serial.println(WiFi.localIP());

  // Web server
  server.on("/", handleRoot);
  server.on("/reset", handleReset);
  server.on("/status", handleStatus);
  server.on("/setIP", handleSetIP);
  server.on("/calibrate", handleCalibrate);
  server.on("/axis", handleAxisConfig);
  server.begin();

  lastTime = millis();
}

void loop()
{
  server.handleClient();

  // ===== Leitura do botão com debounce =====
  int reading = digitalRead(BUTTON_PIN);
  
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading == LOW) {  // Botão pressionado (pull-up = LOW quando pressionado)
      yaw = 0;
      pitch = 0;
      roll = 0;
      Serial.println("Botão pressionado - Orientação resetada!");
      delay(200);  // Pequeno delay para evitar múltiplos resets
    }
  }
  
  lastButtonState = reading;

  unsigned long now = millis();
  dt = (now - lastTime) / 1000.0;
  lastTime = now;

  int16_t data[6];
  if (bmi160.getAccelGyroData(data) == BMI160_OK)
  {

    // Gyro X, Y, Z → Pitch, Roll, Yaw
    float gx = (data[0] - gyroX_offset) / 16.4;
    float gy = (data[1] - gyroY_offset) / 16.4;
    float gz = (data[2] - gyroZ_offset) * (PI / 180.0) / 16.4;

    if (enablePitch) pitch += gx * dt;
    if (enableRoll) roll += gy * dt;
    if (enableYaw) yaw += gz * dt * 180.0 / PI;

    float ax = data[3] / 16384.0;
    float ay = data[4] / 16384.0;
    float az = data[5] / 16384.0;

    // Ângulos pelo acelerômetro
    pitchAcc = atan2(ay, az) * 180.0 / PI;
    rollAcc = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;

    // ===== Filtro complementar =====
    if (enablePitch) pitch = alpha * pitch + (1.0 - alpha) * pitchAcc;
    if (enableRoll) roll = alpha * roll + (1.0 - alpha) * rollAcc;

    if (abs(roll) > 48)
    {
      yaw = 0;
    }

    // ===== Envio UDP =====
    double pkt[6];
    pkt[0] = 0.0;                         // X
    pkt[1] = 0.0;                         // Y
    pkt[2] = 0.0;                         // Z
    pkt[3] = enableYaw ? yaw : 0.0;       // Yaw
    pkt[4] = enablePitch ? pitch : 0.0;   // Pitch
    pkt[5] = enableRoll ? roll : 0.0;     // Roll

    udp.beginPacket(pcIP, pcPort);
    udp.write((uint8_t *)pkt, sizeof(pkt)); // 6 doubles = 48 bytes
    udp.endPacket();
  }

  delay(2);
}
