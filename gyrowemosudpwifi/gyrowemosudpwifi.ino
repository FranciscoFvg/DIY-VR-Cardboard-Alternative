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
}

// ====== WEB ======
void handleRoot()
{
  String html = "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>Head Tracker ESP8266</title>"
                "<script>"
                "setInterval(function(){"
                "fetch('/status').then(r=>r.json()).then(d=>{"
                "document.getElementById('pitch').textContent=d.pitch;"
                "document.getElementById('roll').textContent=d.roll;"
                "document.getElementById('yaw').textContent=d.yaw;"
                "document.getElementById('dest').textContent=d.dest;"
                "});"
                "}, 300);"
                "</script>"
                "</head><body>"
                "<h2>Head Tracker - ESP8266</h2>"
                "<p><b>Pitch:</b> <span id='pitch'>" +
                String(pitch, 2) + "</span></p>"
                                   "<p><b>Roll:</b> <span id='roll'>" +
                String(roll, 2) + "</span></p>"
                                  "<p><b>Yaw:</b> <span id='yaw'>" +
                String(yaw, 2) + "</span></p>"
                                 "<p><b>IP:</b> " +
                WiFi.localIP().toString() + "</p>"
                                            "<p><b>Destino:</b> <span id='dest'>" +
                pcIP.toString() + ":" + String(pcPort) + "</span></p>"
                                                         "<form action='/reset'><button type='submit'>Zerar Yaw</button></form>"
                                                         "<hr>"
                                                         "<form action='/setIP' method='get'>"
                                                         "IP destino: <input name='ip' placeholder='192.168.0.14'><br>"
                                                         "Porta: <input name='port' placeholder='4242'><br>"
                                                         "<button type='submit'>Salvar destino</button>"
                                                         "</form>"
                                                         "</body></html>";

  server.send(200, "text/html", html);
}

void handleReset()
{
  yaw = 0;
  server.sendHeader("Location", "/");
  server.send(303);
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

// ====== SETUP ======
void setup()
{
  Serial.begin(115200);
  delay(2000);

  EEPROM.begin(64); // Inicializa a EEPROM para armazenamento de dados

  // I2C ESP8266
  Wire.begin(D2, D1);

  if (bmi160.I2cInit(i2c_addr) != BMI160_OK)
  {
    Serial.println("Erro ao iniciar BMI160");
    while (1)
      ;
  }

  calibrateGyro();

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
  server.begin();

  lastTime = millis();
}

void loop()
{
  server.handleClient();

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

    pitch += gx * dt;
    roll += gy * dt;
    yaw += gz * dt * 180.0 / PI;

    float ax = data[3] / 16384.0;
    float ay = data[4] / 16384.0;
    float az = data[5] / 16384.0;

    // Ângulos pelo acelerômetro
    pitchAcc = atan2(ay, az) * 180.0 / PI;
    rollAcc = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;

    // ===== Filtro complementar =====
    pitch = alpha * pitch + (1.0 - alpha) * pitchAcc;
    roll = alpha * roll + (1.0 - alpha) * rollAcc;

    if (abs(roll) > 48)
    {
      yaw = 0;
    }

    // ===== Envio UDP =====
    double pkt[6];
    pkt[0] = 0.0;   // X
    pkt[1] = 0.0;   // Y
    pkt[2] = 0.0;   // Z
    pkt[3] = yaw;   // Yaw
    pkt[4] = pitch; // Pitch
    pkt[5] = roll;  // Roll

    udp.beginPacket(pcIP, pcPort);
    udp.write((uint8_t *)pkt, sizeof(pkt)); // 6 doubles = 48 bytes
    udp.endPacket();
  }

  delay(2);
}
