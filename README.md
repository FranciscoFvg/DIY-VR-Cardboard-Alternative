# DIY-VR-Cardboard-Alternative

Projeto pessoal de desenvolvimento de uma alternativa de baixo custo para experimentação de Realidade Virtual usando um smartphone e arduino + sensores.

O projeto surgiu do interesse de experimentar a Realidade Virtual sem desembolsar grandes investimentos, porém com o hardware especializado possuindo um custo muito elevado e sem a possibilidade de uso do smartphone (pela ausência dos sensores necessários).

## Requisitos

Esse projeto utiliza alguns componentes específicos de acordo com a disponibibilidade do autor (eu), mas com algumas adaptações é possível substituir cada um deles por opções equivalentes.

Dito isto, este projeto usa os seguinte componentes:

- Placa chinesa modificada **Wemos D1 R2** equipada com um chip **ESP8266**, com wifi integrado.
- Sensor 6 DOF **BMI160** com acelerometro e giroscópio.
- Google Cardboard (ou outras opções de oculos de realidade virtual para smartphones, como VR BOX).
- Smartphone com Android 8 ou superior.

Além dos componentes de hardware também existem ferramentas de software que devem ser instaladas:

- Arduino IDE (https://www.arduino.cc/en/software/)
- OpenTrack (https://github.com/opentrack/opentrack) - Opcional (modo legado)
- SteamVR (https://store.steampowered.com/app/250820/SteamVR/)
- Parsec (https://parsec.app/downloads) - Opcional (qualquer software de streaming de tela)

> Observação: o projeto também suporta rastreamento direto Arduino → driver OpenVR via Wi-Fi UDP, sem OpenTrack.

## Instalação (WEMOS)

Para utilizar a placa **Wemos D1 R2** em conjunto com o Arduino IDE, deve-se seguir os seguintes passos:

- Baixar o driver USB **CH340** para Windows: <https://sparks.gogo.co.nz/ch340.html>
- Acessar as propriedades do Arduino IDE, ir até a opção "Addicional boards manager URLs" e colar a seguinte URL ( contendo uma nova lista de placas) e clicar em OK:
  - http://arduino.esp8266.com/stable/package_esp8266com_index.json
- Aguardar a conclusão do download da nova lista de placas.
- Abrir a aba de gerenciamento de placas (boards manager), pesquisar por **ESP8266** e instalar a placa.
- Por fim, selecionar a porta COM conectada e atribuir a placa ESP8266->**LOLIN(WEMOS) D1 R2 & mini.**

Quanto ao código, a seguir estão as **_bibliotecas utilizadas_** no Arduino IDE:

- **ESP8266WiFi** (Acompanha o ESP8266 no gerenciamento de placas)
- **ESP8266WebServer** (Acompanha o ESP8266 no gerenciamento de placas)
- **WiFiUdp** (genérica padrão)
- **Wire** (genérica padrão)
- **WiFiManager** (Deve ser instalado na aba gerenciamento de bibliotecas)
- **DFRobot_BMI160** (Deve ser instalado na aba gerenciamento de bibliotecas)

## Configuração

Para conseguir conectar corretamente a placa + sensor, OpenTrack e SteamVR, deve-se realizar alguns passos:

- Instalar o SteamVR através da plataforma Steam.
- Instalar o OpenTrack do site oficial (https://github.com/opentrack/opentrack).
- Instalar o driver que permite a comunicação entre o OpenTrack e o SteamVR. Seguir as instruções do site oficial (https://github.com/r57zone/OpenVR-OpenTrack)
- No OpenTrack, configurar:
  - O input para "UDP over network" mantendo configuração padrão.
  - O output para "freetrack 2.0 Enhanced" mantendo configuração padrão.
  - O filter para "Hamilton" mantendo configuração padrão. (deve ser escolhido ao próprio critério)
  - Mapping mantendo configuração padrão.
  - Em Options, ajustar "Centering method" para "Point".
  - Outras configurações e ajustes podem ser feitas conforme o proprio critério.
