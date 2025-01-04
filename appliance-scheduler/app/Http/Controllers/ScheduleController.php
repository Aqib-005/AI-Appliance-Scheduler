<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class ScheduleController extends Controller
{
    public function index()
    {
        return view('schedule');
    }

    public function store(Request $request)
    {
        set_time_limit(300);

        $request->validate([
            'start_date' => 'required|date',
        ]);

        $response = Http::timeout(120)->post('http://127.0.0.1:8000/predict', [
            'start_date' => $request->input('start_date'),
        ]);

        if ($response->failed()) {
            return back()->withErrors('Failed to fetch predictions. Please try again.');
        }

        $predictions = $response->json()['predictions'];
        return view('results', ['predictions' => $predictions]);
    }
}