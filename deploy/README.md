# Безопасная публикация MusiX через VPS (nginx + обратный SSH-туннель)

## Архитектура

```
 Пользователи ─HTTPS→ VPS (Ubuntu, публичный IP)
                          │  nginx :443  (TLS, rate-limit, прячет /docs)
                          │      ↓ проксирует на
                          │  127.0.0.1:8000  ← дальний конец SSH-туннеля
                          ↑
          обратный SSH ───┘  (домашний сервер сам коннектится к VPS)
                          │
 Домашний сервер ─────────┘
 (Ubuntu, GPU)
   docker compose (musix + qdrant + searxng), musix слушает ТОЛЬКО 127.0.0.1:8000
```

Обе машины — Ubuntu: **домашний сервер** (где крутится приложение + GPU) и
**VPS** (публичный reverse-proxy).

Ключевые свойства:
- На домашнем роутере **не пробрасывается ни один порт**. Домашний сервер сам
  инициирует исходящий SSH к VPS — Роскомнадзор это не блокирует.
- Дома `musix` слушает только loopback → в локалке его не видно, заразить LAN
  через него нельзя.
- Qdrant и SearXNG наружу не выставлены вообще (только внутренняя сеть Compose).
- Наружу у VPS открыты только 22, 80, 443.

---

## Часть 1. Домашний сервер (Ubuntu)

### 1.1. `.env`

В корне репозитория в `.env` задай (сгенерируй секрет!):

```bash
echo "MUSIX_JWT_SECRET=$(openssl rand -hex 32)" >> .env
```
и допиши остальное:
```
PUBLIC_BASE_URL=https://music.example.com
LLM_BASE_URL=...        # если используешь ИИ
OPENAI_API_KEY=...
```

`PUBLIC_BASE_URL` обязателен — иначе ссылки-приглашения соберутся с домашним
LAN-IP и участники не смогут их открыть.

### 1.2. Запуск прод-стека

```bash
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml logs -f musix
```

Проверь, что порт слушается только на loopback (наружу/в LAN не торчит):
```bash
sudo ss -tulpn | grep ':8000'      # ожидается 127.0.0.1:8000, НЕ 0.0.0.0:8000
```

### 1.3. Проверка, что owner уже создан и режим = server

```bash
curl -s http://127.0.0.1:8000/api/v1/instance/config
```

- Ответ вида `{"mode":"server",...}` → **owner создан, bootstrap закрыт** (никто
  снаружи уже не станет владельцем). Именно это и значит «создать owner заранее».
- `mode` должен быть **`server`** — только в нём работают приглашения. Если там
  `sharing`, режим меняется лишь пересозданием инстанса.
- Если вернулось `{"detail":"instance not initialized"}` (404) → owner'а НЕТ,
  создай его ДО открытия наружу (внутри контейнера):
  ```bash
  docker compose -f docker-compose.prod.yml exec musix \
    python -m scripts.create_owner --email you@example.com --password "..." --mode server
  ```

### 1.4. Обратный SSH-туннель (systemd)

Сгенерируй отдельный ключ (без пароля — для автозапуска сервиса):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/musix_tunnel -N "" -C "musix-home-tunnel"
cat ~/.ssh/musix_tunnel.pub          # публичную часть отдашь VPS в шаге 2.2
```

Установи и запусти сервис:
```bash
sudo cp deploy/ssh-tunnel/musix-tunnel.service /etc/systemd/system/
sudo nano /etc/systemd/system/musix-tunnel.service   # впиши User, путь к ключу, VPS_IP, порт
sudo systemctl daemon-reload
sudo systemctl enable --now musix-tunnel
systemctl status musix-tunnel
journalctl -u musix-tunnel -f        # смотреть логи подключения
```

(Опционально) закрой на домашнем сервере вообще весь входящий трафик — туннель
исходящий, ему это не мешает:
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp                # чтобы не потерять локальный SSH к серверу
sudo ufw enable
```

---

## Часть 2. VPS (Ubuntu)

### 2.1. Пакеты

```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx fail2ban ufw
```

### 2.2. Пользователь для туннеля (без shell, только форвардинг)

```bash
sudo adduser --disabled-password --gecos "" tunnel
sudo mkdir -p /home/tunnel/.ssh && sudo chmod 700 /home/tunnel/.ssh
# Вставь СЮДА содержимое deploy/ssh-tunnel/authorized_keys.example,
# подставив свой публичный ключ musix_tunnel.pub:
sudo nano /home/tunnel/.ssh/authorized_keys
sudo chmod 600 /home/tunnel/.ssh/authorized_keys
sudo chown -R tunnel:tunnel /home/tunnel/.ssh
```

В `/etc/ssh/sshd_config` проверь, что разрешён форвардинг (по умолчанию — да):
`AllowTcpForwarding yes`. `GatewayPorts` оставь `no` (порт туннеля привяжется к
127.0.0.1 — именно то, что нужно). Перезапусти: `sudo systemctl restart ssh`.

### 2.3. nginx + TLS

DNS: A-запись `music.example.com` должна указывать на IP VPS ДО выпуска серта.

Наш конфиг ссылается на сертификаты, которых пока нет, поэтому сначала выпускаем
серт через стоковый сайт nginx (:80), и лишь потом включаем свой конфиг:

```bash
# 1) сначала выпускаем сертификат (стоковый сайт nginx отдаёт /var/www/html на :80)
sudo mkdir -p /var/www/html/.well-known/acme-challenge
sudo certbot certonly --webroot -w /var/www/html -d music.example.com

# 2) теперь включаем наш конфиг
sudo cp deploy/nginx/musix.conf /etc/nginx/sites-available/musix.conf
sudo nano /etc/nginx/sites-available/musix.conf      # замени music.example.com
sudo ln -s /etc/nginx/sites-available/musix.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

Автопродление серта certbot ставит сам (таймер systemd); проверить:
`sudo certbot renew --dry-run`.

### 2.4. Файрвол (ufw)

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp      # SSH (или свой порт, если менял)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

Порт **8000 в ufw добавлять НЕ нужно** — он живёт только на 127.0.0.1 и наружу
не смотрит.

### 2.5. fail2ban

```bash
sudo cp deploy/fail2ban/filter.d/musix-auth.conf /etc/fail2ban/filter.d/
sudo cp deploy/fail2ban/jail.d/musix.conf         /etc/fail2ban/jail.d/
sudo systemctl restart fail2ban
sudo fail2ban-client status musix-auth
```

---

## Часть 3. Проверка портов наружу (Ubuntu)

**Что реально слушает и на каком интерфейсе:**
```bash
sudo ss -tulpn
```
Порт `8000` должен быть виден как `127.0.0.1:8000` (loopback), НЕ `0.0.0.0:8000`.
Наружу — только `:22`, `:80`, `:443`.

**Правила и политики файрвола:**
```bash
sudo ufw status verbose      # видно default-политики и разрешённые порты
sudo ufw status numbered
```

**Взгляд снаружи** (с другой машины / телефона по мобильному интернету):
```bash
nmap -Pn -p 22,80,443,8000,6333 <IP_VPS>
```
Ожидается: 80/443 open, 22 open; **8000 и 6333 — closed/filtered**. Если 8000 или
6333 open — что-то настроено не так, наружу не открывай.

**Проверь заодно домашний роутер снаружи** (что ты случайно ничего не пробросил):
```bash
nmap -Pn -p 8000,6333,8088 <публичный_IP_дома>
```
Всё должно быть closed/filtered.

---

## Часть 4. Выдача ключей доступа (приглашения)

Инстанс в режиме `server` — регистрация только по одноразовому коду.

Owner создаёт приглашение (получив свой JWT через `/api/v1/auth/login`):
```bash
TOKEN=<jwt owner'а>
curl -s -X POST https://music.example.com/api/v1/auth/invites \
     -H "Authorization: Bearer $TOKEN"
```
В ответе — `link` (готовая ссылка регистрации) и `code`. Отдаёшь человеку ссылку;
он регистрируется email + пароль + код. Код **одноразовый, живёт 7 дней**.

Это удобнее делать из веб-интерфейса owner'а (раздел управления инвайтами) — там
те же вызовы под капотом.

> Напоминание: у каждого участника **своя изолированная библиотека**. Твою музыку
> они не видят — это by design.

---

## Чеклист перед открытием наружу

- [ ] `MUSIX_JWT_SECRET` задан (64 hex), dev-фоллбэка в логах нет.
- [ ] `/api/v1/instance/config` отдаёт `mode: server` (owner создан заранее).
- [ ] `docker compose -f docker-compose.prod.yml` поднят; `ss -tulpn` дома
      показывает `127.0.0.1:8000`, Qdrant/SearXNG хост-портов нет.
- [ ] SSH-туннель работает и автозапускается.
- [ ] nginx отдаёт сайт по HTTPS; `/docs` возвращает 404.
- [ ] `ufw status` — открыты только 22/80/443; `nmap` снаружи подтверждает.
- [ ] fail2ban активен (`fail2ban-client status`).
- [ ] `nmap` домашнего публичного IP — 8000/6333/8088 закрыты.
