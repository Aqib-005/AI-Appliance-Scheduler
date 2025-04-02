<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\ScheduleController;
use Illuminate\Support\Facades\DB;

// Dashboard
Route::get('/', [ScheduleController::class, 'dashboard'])->name('dashboard');

Route::post('/schedule', [ScheduleController::class, 'store'])->name('schedule.store');
// Schedule Page
Route::get('/schedule/create', [ScheduleController::class, 'createSchedule'])->name('schedule.create');
Route::post('/schedule/store', [ScheduleController::class, 'storeSchedule'])->name('schedule.store');
Route::get('/results', [ScheduleController::class, 'showResults'])->name('results');
Route::post('/selected-appliance/add', [ScheduleController::class, 'addSelectedAppliance'])->name('selected-appliance.add');
Route::get('/selected-appliance/get/{id}', [ScheduleController::class, 'getSelectedAppliance']);
Route::put('/selected-appliance/update/{id}', [ScheduleController::class, 'updateSelectedAppliance']);
Route::delete('/selected-appliance/remove/{id}', [ScheduleController::class, 'removeSelectedAppliance'])->name('selected-appliance.remove');


// Manage Appliances
Route::get('/appliances/manage', [ScheduleController::class, 'manageAppliances'])->name('appliances.manage');
Route::post('/appliance/add', [ScheduleController::class, 'addAppliance'])->name('appliance.add');
Route::put('/appliance/edit/{id}', [ScheduleController::class, 'editAppliance'])->name('appliance.edit');
Route::put('/appliance/update/{id}', [ScheduleController::class, 'updateAppliance'])->name('appliance.update');
Route::delete('/appliance/remove/{id}', [ScheduleController::class, 'removeAppliance'])->name('appliance.remove');
Route::get('/appliance/get/{id}', [ScheduleController::class, 'getAppliance'])->name('appliance.get');