# ⚡ AI-Powered Appliance Scheduler

This project is a smart appliance scheduling system that leverages machine learning to predict electricity prices and optimize appliance usage for cost-efficiency. It was developed as part of my final-year university dissertation.

---

## 🎯 Purpose

The goal is to help households reduce electricity bills by automatically scheduling appliances to run during cheaper time slots — based on predicted day-ahead market electricity prices.

---

## 🧠 Core Features

- 🔮 **Electricity Price Prediction**  
  Uses a hybrid ML model (Random Forest + LSTM) to perform day-ahead and weekly electricity price forecasting.

- 📅 **Cost-Efficient Scheduling**  
  Implements a greedy scheduling algorithm that allocates appliance run-times within preferred user time windows to minimize cost.

- 🧾 **User Input Modeling**  
  Users define appliances, energy usage, duration, and time preferences via a clean web interface.

- 📊 **Result Visualization**  
  Displays price trends, scheduled time slots, and projected cost savings via charts and summary tables.

---

## 🏗️ Architecture Overview

```text
[User Input (Frontend: Laravel)]
        │
        ▼
[Scheduler Logic (PHP)]
        │
        ├──> Appliance Preference Parsing
        ├──> Greedy Algorithm Cost Minimization
        ▼
[Predicted Prices (Python Model)]
        │
        ├──> Data Preprocessing
        ├──> RF-LSTM Hybrid Inference
        ▼
[Database (SQL)]
        │
        └──> Stores predictions, schedules, usage logs
```
---

## 📖 Instructions to run Appliance Scheduler

## Prerequisites

Make sure you have python, php, laravel and mysql installed from their offical websites

## Web-Interface Setup:

1. Install PHP dependencies:

```bash
composer install
```

2. Set up environment config:

```bash
copy .env.example .env
```

```bash
php artisan key:generate
```

3. Start Laravel Server:

```bash
php artisan serve
```

The Web Interface will run at: http://127.0.0.1:8000

## Databse setup:

Because MySQL is hosted locally you must create the database on your own system and configure the .env file with your own username/password. The Laravel app will not work without this step.

1. Create the database manually (via phpMyAdmin or CLI):

```bash
CREATE DATABASE appliance_scheduler;
```

2. Edit .env to match your local MySQL credentials:

```bash
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=appliance_scheduler
DB_USERNAME=root
DB_PASSWORD=
```

3. Run migrations:

```bash
php artisan migrate
```

4. Import prebuilt schema:

```bash
mysql -u root -p appliance_scheduler < database.sql
```

## Python ML API Setup:

1. Go to ml_model directory:

```bash
cd appliance-scheduler

cd ml_model
```

2. Create virtual env:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install libraries:

```bash
pip install -r requirements.txt
```

4. Run ML API server:

```bash
uvicorn api:app --reload --port 8001
```

The FastAPI ML model will run at: http://127.0.0.1:8001

Both the web interface and the python API must be running at the same time.
