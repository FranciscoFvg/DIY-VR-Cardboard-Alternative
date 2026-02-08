#include <ESP8266WiFi.h> // Uso da placa WEMOS D1 R2
#include <ESP8266WebServer.h> // Servidor web
#include <WiFiUdp.h> // Envio de pacotes UDP
#include <DFRobot_BMI160.h> // Comunicação com sensor 6 DOF BMI160
#include <Wire.h> // Conexão I2C
#include <WiFiManager.h> // Permite configuração do wifi sem regravação

DFRobot_BMI160 bmi160; 
WiFiUDP udp;
ESP8266WebServer server(80);

// ====== OPENTRACK (PC) ======
IPAddress pcIP(192,168,0,14); // IP da máquina com OPENTRACK
const unsigned int pcPort = 4242; // porta usada pelo OPENTRACK

// ====== BMI160 ======
const int8_t i2c_addr = 0x69;

float ax, ay, az;
float pitch, roll, yaw = 0;

float gyroX_offset = 0;
float gyroY_offset = 0;
float gyroZ_offset = 0;

unsigned long lastTime;
float dt;

struct FreeTrackPacket {
  uint32_t header;
  uint32_t frame;
  float yaw;
  float pitch;
  float roll;
  float x;
  float y;
  float z;
};

uint32_t frameCounter = 0;

// ====== CALIBRAÇÃO ======
void calibrateGyro() {
  const int samples = 500;
  long sumX = 0, sumY = 0, sumZ = 0;

  Serial.println("Calibrando Gyro X, Y e Z... NÃO MEXA O SENSOR");
  delay(1000);

  for (int i = 0; i < samples; i++) {
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

  Serial.print("Offset Gyro X = "); Serial.println(gyroX_offset);
  Serial.print("Offset Gyro Y = "); Serial.println(gyroY_offset);
  Serial.print("Offset Gyro Z = "); Serial.println(gyroZ_offset);
}

// ====== WEB ======
void handleRoot() {
  String html = "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<meta http-equiv='refresh' content='0.2'>"
                "<title>Head Tracker ESP8266</title></head><body>"
                "<h2>Head Tracker - ESP8266</h2>"
                "<p><b>Pitch:</b> " + String(pitch, 2) + "</p>"
                "<p><b>Roll:</b> "  + String(roll,  2) + "</p>"
                "<p><b>Yaw:</b> "   + String(yaw,   2) + "</p>"
                "<p><b>IP:</b> "    + WiFi.localIP().toString() + "</p>"
                "<form action='/reset'><button type='submit'>Zerar Yaw</button></form>"
                "</body></html>";

  server.send(200, "text/html", html);
}

void handleReset() {
  yaw = 0;
  server.sendHeader("Location", "/");
  server.send(303);
}

// ====== SETUP ======
void setup() {
  Serial.begin(115200);
  delay(2000);

  // I2C ESP8266
  Wire.begin(D2, D1);

  if (bmi160.I2cInit(i2c_addr) != BMI160_OK) {
    Serial.println("Erro ao iniciar BMI160");
    while (1);
  }

  calibrateGyro();

  // ===== WiFiManager =====
  WiFiManager wm;
  wm.setConfigPortalTimeout(180);

  if (!wm.autoConnect("HeadTracker-ESP8266")) {
    Serial.println("Falha ao conectar. Reiniciando...");
    ESP.restart();
  }
  
  Serial.println("\nConectado!");
  Serial.print("IP do ESP: ");
  Serial.println(WiFi.localIP());

  // Web server
  server.on("/", handleRoot);
  server.on("/reset", handleReset);
  server.begin();

  lastTime = millis();
}

void loop() {
  server.handleClient();

  unsigned long now = millis();
  dt = (now - lastTime) / 1000.0;
  lastTime = now;

  int16_t data[6];
  if (bmi160.getAccelGyroData(data) == BMI160_OK) {

    // Gyro X, Y, Z → Pitch, Roll, Yaw
    float gx = (data[0] - gyroX_offset) * (PI / 180.0) / 16.4;
    float gy = (data[1] - gyroY_offset) / 16.4;
    float gz = (data[2] - gyroZ_offset) * (PI / 180.0) / 16.4;

    pitch += gx * dt * 180.0 / PI;
    roll  += gy * dt;
    yaw   += gz * dt * 180.0 / PI;

    // ===== Envio UDP =====
    double pkt[6];
    pkt[0] = 0.0;    // X
    pkt[1] = 0.0;    // Y
    pkt[2] = 0.0;    // Z
    pkt[3] = yaw;    // Yaw
    pkt[4] = pitch;  // Pitch
    pkt[5] = roll;   // Roll

    udp.beginPacket(pcIP, pcPort);
    udp.write((uint8_t*)pkt, sizeof(pkt));  // 6 doubles = 48 bytes
    udp.endPacket();

  }

  delay(10);
}
