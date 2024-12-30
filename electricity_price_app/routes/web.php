<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\PredictionController;

Route::get('/', [PredictionController::class, 'index']);
Route::post('/schedule', [PredictionController::class, 'schedule']);

// Route::get('/', function () {
//     return view('welcome');
// });
