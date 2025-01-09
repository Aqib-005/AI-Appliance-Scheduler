<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use App\Models\Appliance;
use App\Models\Schedule;

class ScheduleController extends Controller
{
    // Display the dashboard
    public function dashboard()
    {
        // Fetch appliances and schedules
        $appliances = Appliance::all();
        $schedules = Schedule::with('appliance')->get();
        $predictions = []; // Fetch predictions from your AI model (if needed)

        return view('dashboard', [
            'appliances' => $appliances,
            'schedules' => $schedules,
            'predictions' => $predictions,
        ]);
    }

    // Fetch predictions and generate the schedule
    public function store(Request $request)
    {
        set_time_limit(300);

        // Log the start_date being sent to FastAPI
        \Log::info('Sending start_date to FastAPI', ['start_date' => $request->input('start_date')]);

        // Call the FastAPI endpoint to get predictions
        $response = Http::timeout(300)->post('http://127.0.0.1:8000/predict', [
            'start_date' => $request->input('start_date'), // Ensure this is in the correct format
        ]);

        // Log the response from FastAPI
        \Log::info('FastAPI response', ['status' => $response->status(), 'body' => $response->body()]);

        if ($response->failed()) {
            return back()->withErrors('Failed to fetch predictions. Please try again.');
        }

        // Get the predictions
        $predictions = $response->json()['predictions'];

        // Retrieve appliances from the database
        $appliances = Appliance::all()->map(function ($appliance) {
            return [
                'name' => $appliance->name,
                'power' => $appliance->power,
                'preferred_start' => $appliance->preferred_start,
                'preferred_end' => $appliance->preferred_end,
                'duration' => $appliance->duration,
                'usage_days' => json_decode($appliance->usage_days, true) ?? [], // Default to an empty array if null
            ];
        })->toArray();

        // Schedule the appliances
        $schedule = $this->scheduleAppliances($predictions, $appliances);

        // Pass the schedule to the view
        return view('results', ['predictions' => $predictions, 'schedule' => $schedule]);
    }

    // Schedule appliances based on predictions
    private function scheduleAppliances($predictions, $appliances)
    {
        set_time_limit(300);

        // Sort predictions by price (ascending)
        usort($predictions, function ($a, $b) {
            return $a['Predicted Price [Euro/MWh]'] <=> $b['Predicted Price [Euro/MWh]'];
        });

        // Initialize the schedule
        $schedule = [];

        // Group predictions by day
        $predictionsByDay = [];
        foreach ($predictions as $prediction) {
            $day = date('l', strtotime($prediction['Start date/time'])); // Get the day name (e.g., Monday)
            $hour = (int) date('H', strtotime($prediction['Start date/time'])); // Get the hour (0-23)
            $predictionsByDay[$day][$hour] = $prediction; // Store predictions by day and hour
        }

        // Sort appliances by priority (higher power consumption × duration first)
        usort($appliances, function ($a, $b) {
            $priorityA = $a['power'] * $a['duration'];
            $priorityB = $b['power'] * $b['duration'];
            return $priorityB <=> $priorityA; // Sort in descending order
        });

        // Assign appliances to the cheapest hours within their preferred time slots
        foreach ($appliances as $appliance) {
            foreach ($appliance['usage_days'] as $day) {
                if (!isset($predictionsByDay[$day])) {
                    continue; // Skip if no predictions for this day
                }

                // Find all possible windows within the preferred time range
                $windows = [];
                for ($startHour = $appliance['preferred_start']; $startHour + $appliance['duration'] - 1 <= $appliance['preferred_end']; $startHour++) {
                    // Calculate the total cost for this window
                    $totalCost = 0;
                    $conflict = false;

                    for ($i = 0; $i < $appliance['duration']; $i++) {
                        $currentHour = $startHour + $i;

                        // Check if the hour is available
                        if (isset($schedule[$day][$currentHour])) {
                            $conflict = true;
                            break;
                        }

                        // Add the price for this hour
                        if (isset($predictionsByDay[$day][$currentHour])) {
                            $totalCost += $predictionsByDay[$day][$currentHour]['Predicted Price [Euro/MWh]'];
                        } else {
                            $conflict = true; // Skip if any hour in the window is missing
                            break;
                        }
                    }

                    if (!$conflict) {
                        $windows[] = [
                            'startHour' => $startHour,
                            'totalCost' => $totalCost,
                        ];
                    }
                }

                // Sort windows by total cost (ascending)
                usort($windows, function ($a, $b) {
                    return $a['totalCost'] <=> $b['totalCost'];
                });

                // Assign the appliance to the cheapest available window
                if (!empty($windows)) {
                    $cheapestWindow = $windows[0];
                    $startHour = $cheapestWindow['startHour'];

                    for ($i = 0; $i < $appliance['duration']; $i++) {
                        $currentHour = $startHour + $i;
                        $schedule[$day][$currentHour] = [
                            'appliance' => $appliance['name'],
                            'power' => $appliance['power'],
                            'hour' => $currentHour,
                            'price' => $predictionsByDay[$day][$currentHour]['Predicted Price [Euro/MWh]'],
                        ];
                    }
                }
            }
        }

        return $schedule;
    }

    public function getAppliance($id)
    {
        $appliance = Appliance::findOrFail($id);
        return response()->json([
            'id' => $appliance->id,
            'name' => $appliance->name,
            'preferred_start' => $appliance->preferred_start,
            'preferred_end' => $appliance->preferred_end,
            'duration' => $appliance->duration,
        ]);
    }

    // Add a new appliance
    public function addAppliance(Request $request)
    {
        $request->validate([
            'name' => 'required|string',
            'power' => 'required|numeric',
            'preferred_start' => 'required|integer|min:0|max:23',
            'preferred_end' => 'required|integer|min:0|max:23',
            'duration' => 'required|numeric',
        ]);

        Appliance::create([
            'name' => $request->input('name'),
            'power' => $request->input('power'),
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
            'usage_days' => null, // Set usage_days to null
        ]);

        return redirect()->route('appliances.manage')->with('success', 'Appliance added successfully.');
    }

    // Remove an appliance
    public function removeAppliance($id)
    {
        $appliance = Appliance::findOrFail($id);
        $appliance->delete();

        return redirect()->back()->with('success', 'Appliance removed successfully.');
    }

    // Manage appliances (view all)
    public function manageAppliances()
    {
        $appliances = Appliance::all();
        return view('manage', ['appliances' => $appliances]);
    }

    // Edit an appliance
    public function editAppliance(Request $request, $id)
    {
        $request->validate([
            'name' => 'required|string',
            'power' => 'required|numeric',
            'preferred_start' => 'required|integer|min:0|max:23',
            'preferred_end' => 'required|integer|min:0|max:23',
            'duration' => 'required|numeric',
        ]);

        $appliance = Appliance::findOrFail($id);
        $appliance->update([
            'name' => $request->input('name'),
            'power' => $request->input('power'),
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
            'usage_days' => null, // Set usage_days to null
        ]);

        return redirect()->route('appliances.manage')->with('success', 'Appliance updated successfully.');
    }

    // Update an appliance
    public function updateAppliance(Request $request, $id)
    {
        $request->validate([
            'preferred_start' => 'required|date_format:H:i', // Validate time format (HH:mm)
            'preferred_end' => 'required|date_format:H:i',   // Validate time format (HH:mm)
            'duration' => 'required|numeric',                // Validate duration as a decimal
        ]);

        $appliance = Appliance::findOrFail($id);
        $appliance->update([
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'), // Store duration as a decimal
        ]);

        return response()->json(['success' => true]);
    }

    public function createSchedule()
    {
        $appliances = Appliance::all();
        return view('schedule', ['appliances' => $appliances]);
    }
}