# FYProject

Dissertation

AI API:
uvicorn api:app --reload --port 8000

Laravel App:
php artisan serve --port=8001

TODO:

1. Make three models ✔
2. Select best model ✔
3. Use the model to get predcitons for next 7 days
   a. got it to predict 7 days but get it accuarte (✔️ kinda using XGBoost for now)
4. Store predcition into cache ✔️
5. Start laravel...
6. made the UI stuff
   a. Appliaces
   a1. need to fix edit applaince button ✔️
   1a1. validate edit/add inputs ❌
   b. Scheduler
   Need to ensure that the user can hard select applonace for a set time
   c. View Schedule/DashBoard
