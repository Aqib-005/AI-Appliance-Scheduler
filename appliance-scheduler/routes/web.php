<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\ScheduleController;
use Illuminate\Support\Facades\DB;

// Dashboard
Route::get('/', [ScheduleController::class, 'dashboard'])->name('dashboard');

Route::get('/appliances/manage', [ScheduleController::class, 'manageAppliances'])->name('appliances.manage');

Route::post('/appliance/add', [ScheduleController::class, 'addAppliance'])->name('appliance.add');

Route::get('/appliance/edit/{id}', [ScheduleController::class, 'editAppliance'])->name('appliance.edit');
Route::put('/appliance/update/{id}', [ScheduleController::class, 'updateAppliance'])->name('appliance.update');

Route::delete('/appliance/remove/{id}', [ScheduleController::class, 'removeAppliance'])->name('appliance.remove');

Route::get('/test-db', function () {
    try {
        DB::connection()->getPdo();
        return "Connected successfully to the database!";
    } catch (\Exception $e) {
        return "Could not connect to the database. Error: " . $e->getMessage();
    }
});

Route::get('/schedule', [ScheduleController::class, 'index'])->name('schedule.index');
Route::post('/schedule', [ScheduleController::class, 'store'])->name('schedule.store');
