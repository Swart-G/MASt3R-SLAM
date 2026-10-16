# MASt3R-SLAM

Плотная 3D-реконструкция пространства по кадрам с Android-камеры, видео и наборам изображений.

Этот форк дополняет исследовательский MASt3R-SLAM мобильным сканером, сохранением
проектов и двумя интерфейсами: **TUI в терминале** и **Web Panel в браузере**.
Оба интерфейса используют общий движок и один формат проектов. Веб-панель
запускается через Docker Compose, вычисления выполняются на NVIDIA GPU компьютера.

[Статья](https://arxiv.org/abs/2412.12392) · [Видео](https://youtu.be/wozt71NBFTQ) · [Исходный проект](https://edexheim.github.io/mast3r-slam/)

## Содержание

- [Возможности](#возможности)
- [Стек](#стек)
- [Установка и запуск](#установка-и-запуск)
- [Настройки](#настройки)
- [Видео, изображения и RealSense](#видео-изображения-и-realsense)
- [API](#api)
- [Документация](#документация)
- [Проверки](#проверки)
- [Авторы и благодарности](#авторы-и-благодарности)
- [Лицензия](#лицензия)

## Возможности

- Два способа управления: терминальный интерфейс Textual и адаптивная веб-панель на Bootstrap.
- Подключение Android-приложения по QR-коду с токеном сопряжения и отпечатком TLS-сертификата.
- Предпросмотр обработанного кадра и компактное состояние сканирования.
- Пауза, продолжение и завершение сканирования с сохранением реконструкции.
- Библиотека проектов, контрольные точки, ревизии и продолжение после релокализации.
- Экспорт PLY для последующей работы в инструментах просмотра облаков точек.
- Подготовка мобильного облака и AR-просмотр в Android-клиенте с совместимым ARCore-устройством.
- Исходный режим обработки видео, папок RGB-изображений, датасетов и Intel RealSense.

Мобильное сканирование требует установленного Android-приложения и доступного
по локальной сети Linux-компьютера с NVIDIA GPU. Предпросмотр в веб-панели —
обработанный JPEG из движка; готовую геометрию можно скачать как PLY.

## Стек

| Часть проекта | Технологии |
| --- | --- |
| Реконструкция | Python 3.11, PyTorch, CUDA, MASt3R, Lietorch, собственные C++/CUDA-расширения |
| Терминальный интерфейс | Textual, Rich |
| Веб-панель | aiohttp, HTML, CSS, JavaScript, Bootstrap 5.3.8 |
| Android-клиент | CameraX, OpenGL ES, ARCore |
| Обмен данными | TLS, WebSocket, Protobuf, mDNS |
| Хранение | Локальные проекты, JSON-манифесты, NPZ-контрольные точки, PLY |
| Сборка и запуск | Docker, Docker Compose, uv |
| Проверки | unittest, aiohttp TestClient, Textual Pilot |

Ресурсы веб-интерфейса поставляются локально. Интернет нужен при сборке образа
и загрузке зависимостей и весов; для отображения панели CDN не требуется.

## Установка и запуск

### Вариант 1 — Web Panel через Docker Compose

Нужны **Linux x86-64**, Docker Engine, Docker Compose v2+, драйвер NVIDIA
и NVIDIA Container Toolkit. Сборка использует CUDA 12.8 и PyTorch 2.9.1;
драйвер должен поддерживать этот CUDA runtime. CUDA Toolkit на хосте для
контейнерной сборки устанавливать не нужно.

Из корня этого репозитория:

```bash
# Получить исходники вложенных зависимостей
git submodule update --init --recursive

# Загрузить веса с проверкой контрольных сумм
mkdir -p projects checkpoints
python3 desktop/download_checkpoints.py checkpoints

# Подготовить настройки и запустить панель
cp .env.example .env
docker compose up -d --build --wait
```

Первая сборка скачивает CUDA-образ, PyTorch и собирает нативные расширения.
Она требует времени и свободного места. Веса хранятся отдельно от образа.

| Назначение | Адрес по умолчанию |
| --- | --- |
| Веб-панель | [http://127.0.0.1:8080](http://127.0.0.1:8080) |
| Готовность HTTP-сервера | [http://127.0.0.1:8080/health](http://127.0.0.1:8080/health) |
| Подключение Android | TLS-порт `8765` на LAN-адресе компьютера; адрес находится в QR-коде |

Compose использует сеть хоста Linux. Панель доступна локально на компьютере,
а телефон подключается к отдельному TLS-серверу по локальной сети.
`/health` проверяет HTTP-сервер; готовность GPU и весов определяется при работе движка.

**Первое сканирование:**

1. Откройте панель, нажмите «Новый сеанс», введите название проекта и нажмите «Создать».
2. На Android откройте приложение MASt3R-SLAM и отсканируйте QR-код.
3. Запустите сканирование в приложении. Новый сеанс загружает модель при первом кадре.
4. Для сохранения нажмите «Завершить» в панели или на телефоне.
5. В разделе «Мои проекты» скачайте PLY или продолжите реконструкцию.

Сборка Android-клиента описана в [android-app/README.md](android-app/README.md).
Локально собранные APK и Linux-лаунчер размещаются в [releases/](releases/README.md).

Обновить приложение и посмотреть логи:

```bash
docker compose up -d --build --wait
docker compose logs -f web
```

Остановить контейнер:

```bash
docker compose down
```

Проекты остаются в `projects/` на компьютере. Остановка Compose их не удаляет.
Если `id -u` или `id -g` отличаются от `1000`, задайте соответствующие
`MAST3R_UID` и `MAST3R_GID` в `.env`, чтобы TUI мог работать с теми же файлами.

### Вариант 2 — TUI в терминале

Для готового Linux-лаунчера используйте инструкции в [releases/README.md](releases/README.md).
Для запуска из исходников нужны Python 3.11, компилятор C++, NVIDIA GPU,
драйвер и совместимый с PyTorch CUDA Toolkit (`nvcc`).

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install setuptools==70.0.0 wheel ninja numpy==1.26.4

git submodule update --init --recursive
python desktop/prepare_build_environment.py .
python -m pip install --no-build-isolation -e thirdparty/mast3r thirdparty/in3d -e .
python desktop/download_checkpoints.py checkpoints

mast3r-slam tui
```

Для приведённого примера нужен CUDA Toolkit 12.8. Скрипт подготовки скачивает
закреплённый Lietorch, настраивает Eigen и исправляет заголовок PyTorch для сборки.

Запуск без аргументов также открывает TUI:

```bash
mast3r-slam
```

В установленном окружении доступны и другие команды:

```bash
mast3r-slam web                       # Локальная веб-панель без контейнера
mast3r-slam serve                     # Сервер Android без интерфейса
mast3r-slam projects                  # Список сохранённых проектов
mast3r-slam continue <project-id>      # Продолжить проект через сервер Android
mast3r-slam export <project-id> out.ply

# Необязательный desktop-просмотрщик
python -m pip install '.[viewer]'
mast3r-slam view <project-id>
```

Одновременно используйте один интерфейс для одной папки проектов:
TUI и Web используют одинаковые проекты и порт телефона.

## Настройки

Параметры Compose задаются в `.env`; пример — [.env.example](.env.example).

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `MAST3R_WEB_PORT` | `8080` | Локальный HTTP-порт панели |
| `MAST3R_PHONE_PORT` | `8765` | TLS-порт Android-сервера |
| `MAST3R_UID` / `MAST3R_GID` | `1000` / `1000` | UID и GID пользователя хоста для записи проектов |
| `TORCH_CUDA_ARCH_LIST` | `8.0;8.6;8.9;9.0;12.0+PTX` | Архитектуры GPU для контейнерной сборки |
| `MAST3R_SLAM_PROJECTS` | `projects` | Папка проектов для локального запуска; в Compose — `/app/projects` |

Веса монтируются из `checkpoints/`, конфигурация — из `config/`, оба каталога
в контейнере доступны только для чтения. Для ускорения сборки можно оставить
в `TORCH_CUDA_ARCH_LIST` архитектуру своей GPU: например, `12.0` для RTX 50xx.

```bash
mast3r-slam --projects-dir /path/to/projects tui
mast3r-slam web --port 8090 --phone-port 8766 --config config/base.yaml
```

## Видео, изображения и RealSense

Исходный pipeline доступен через `main.py` и команду `legacy`:

```bash
# Видео или папка RGB-изображений
python main.py --dataset /path/to/video.mp4 --config config/base.yaml
python main.py --dataset /path/to/images --config config/base.yaml

# Известная калибровка камеры
python main.py --dataset /path/to/video.mp4 --config config/base.yaml \
  --calib config/intrinsics.yaml

# Подключённая камера Intel RealSense
python main.py --dataset realsense --config config/base.yaml

# Пример TUM-RGBD
bash scripts/download_tum.sh
mast3r-slam legacy -- --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```

Для оценки на датасетах есть `scripts/eval_tum.sh`, `eval_7_scenes.sh`,
`eval_euroc.sh`, `eval_eth3d.sh` и соответствующие скрипты загрузки.
Производительность зависит от GPU; авторы исходного проекта проводили эксперименты на RTX 4090.

## API

Веб-панель предоставляет HTTP API для проектов и управления сеансом.
Основные методы: `GET /api/status`, `GET /api/projects`, `POST /api/session`,
`POST /api/commands/pause`, `POST /api/commands/resume`, `POST /api/commands/stop`.

Полная таблица маршрутов и требования к запросам — в [описании веб-панели](docs/web-panel.md#http-api).
Android использует отдельный версионируемый [Protobuf-контракт](protocol/README.md).

## Документация

| Документ | Содержание |
| --- | --- |
| [Веб-панель](docs/web-panel.md) | Дизайн, Compose, сценарии работы и HTTP API |
| [Мобильный сканер](docs/mobile-scanner.md) | Архитектура, формат проектов и ограничения проверки |
| [Android](android-app/README.md) | Сборка и устройство приложения |
| [Desktop](desktop/README.md) | Команды и Linux-лаунчер |
| [Протокол](protocol/README.md) | Контракт Android ↔ Linux |
| [Релизные артефакты](releases/README.md) | APK, Linux-лаунчер и контрольные суммы |
| [Исходные зависимости](Dependencies.md) | Зависимости исследовательского проекта |

## Проверки

```bash
source venv/bin/activate
python -m unittest discover -s tests
docker compose config -q
```

Автоматические проверки покрывают ядро, протокол, проекты, TUI и HTTP API
панели. Они не заменяют сквозное сканирование с реальным Android-устройством и GPU.

## Авторы и благодарности

Исходный MASt3R-SLAM разработали [Riku Murai](https://rmurai.co.uk/),
[Eric Dexheimer](https://edexheim.github.io/) и [Andrew J. Davison](https://www.doc.ic.ac.uk/~ajd/).
Murai и Dexheimer внесли равный вклад.

Проект использует [MASt3R](https://github.com/naver/mast3r),
[MASt3R-SfM](https://github.com/naver/mast3r/tree/mast3r_sfm),
[DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) и
[ModernGL](https://github.com/moderngl/moderngl).
Структура этого README вдохновлена [ServiceOil](https://github.com/Krastti/ServiceOil).

Если вы используете MASt3R-SLAM в исследовании, укажите исходную работу:

```bibtex
@inproceedings{murai2024_mast3rslam,
  title={{MASt3R-SLAM}: Real-Time Dense {SLAM} with {3D} Reconstruction Priors},
  author={Murai, Riku and Dexheimer, Eric and Davison, Andrew J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```

## Лицензия

Проект распространяется по **CC BY-NC-SA 4.0**. Условия использования и
обязательные уведомления о сторонних компонентах собраны в [LICENSE.md](LICENSE.md).
