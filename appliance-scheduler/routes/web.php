<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\ScheduleController;
use Illuminate\Support\Facades\DB;

// Dashboard
Route::get('/', [ScheduleController::class, 'dashboard'])->name('dashboard');

// Schedule Page
Route::get('/schedule/create', [ScheduleController::class, 'createSchedule'])->name('schedule.create');
Route::post('/schedule/store', [ScheduleController::class, 'storeSchedule'])->name('schedule.store');

// Manage Appliances
Route::get('/appliances/manage', [ScheduleController::class, 'manageAppliances'])->name('appliances.manage');
Route::post('/appliance/add', [ScheduleController::class, 'addAppliance'])->name('appliance.add');
Route::get('/appliance/edit/{id}', [ScheduleController::class, 'editAppliance'])->name('appliance.edit');
Route::put('/appliance/update/{id}', [ScheduleController::class, 'updateAppliance'])->name('appliance.update');
Route::delete('/appliance/remove/{id}', [ScheduleController::class, 'removeAppliance'])->name('appliance.remove');
