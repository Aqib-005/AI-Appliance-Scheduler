<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use App\Models\Appliance;
use App\Models\SelectedAppliance;
use App\Models\Schedule;

class ScheduleController extends Controller
{
    // Display the dashboard
    public function dashboard()
    {
        // Fetch appliances and selected appliances with the appliance relationship
        $appliances = Appliance::all();
        $selectedAppliances = SelectedAppliance::with('appliance')->get();

        // Debugging: Check for null appliance relationships
        foreach ($selectedAppliances as $selectedAppliance) {
            if (!$selectedAppliance->appliance) {
                \Log::warning('SelectedAppliance has no associated Appliance', [
                    'selected_appliance_id' => $selectedAppliance->id,
                    'appliance_id' => $selectedAppliance->appliance_id,
                ]);
            }
        }

        // Fetch predictions from the session or database
        $predictions = session('predictions', []); // Retrieve from session or default to an empty array

        // Pass the data to the view
        return view('dashboard', [
            'appliances' => $appliances,
            'selectedAppliances' => $selectedAppliances,
            'predictions' => $predictions,
        ]);
    }

    public function showResults()
    {
        // Retrieve the schedule and predictions from the session or database
        $schedule = session('schedule'); // Example: Retrieve from session
        $predictions = session('predictions'); // Example: Retrieve from session

        // Pass the data to the view
        return view('results', [
            'schedule' => $schedule,
            'predictions' => $predictions,
        ]);
    }

    // Fetch predictions and generate the schedule
    public function store(Request $request)
    {
        set_time_limit(300);

        // Calculate the most recent Monday from the system time
        $start_date = date('Y-m-d', strtotime('monday this week'));

        // Log the calculated start_date
        \Log::info('Calculated start_date:', ['start_date' => $start_date]);

        // Call the FastAPI endpoint to get predictions
        try {
            $response = Http::timeout(300)->post('http://127.0.0.1:8000/predict', [
                'start_date' => $start_date,
            ]);

            // Log the response from FastAPI
            \Log::info('FastAPI response', [
                'status' => $response->status(),
                'body' => $response->body(),
            ]);

            if ($response->failed()) {
                \Log::error('FastAPI request failed', [
                    'status' => $response->status(),
                    'body' => $response->body(),
                ]);
                return response()->json(['success' => false, 'message' => 'Failed to fetch predictions. Please try again.'], 500);
            }

            // Get the predictions
            $predictions = $response->json()['predictions'];

            // Retrieve appliances from the database
            $appliances = Appliance::all()->map(function ($appliance) {
                return [
                    'id' => $appliance->id,
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

            // Redirect to the dashboard
            return response()->json([
                'success' => true,
                'redirect_url' => route('dashboard'), // Redirect to the dashboard
            ]);
        } catch (\Exception $e) {
            \Log::error('Error in store method:', ['error' => $e->getMessage()]);
            return response()->json(['success' => false, 'message' => 'An error occurred. Please try again.'], 500);
        }
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

                    // Calculate predicted start and end times
                    $predictedStartTime = date('H:i', strtotime("$startHour:00"));
                    $predictedEndTime = date('H:i', strtotime("$startHour:00") + ($appliance['duration'] * 3600));

                    // Save the predicted times to the selected_appliances table
                    SelectedAppliance::where('appliance_id', $appliance['id'])
                        ->whereJsonContains('usage_days', strtolower($day))
                        ->update([
                            'predicted_start_time' => $predictedStartTime,
                            'predicted_end_time' => $predictedEndTime,
                        ]);

                    // Add to the schedule
                    for ($i = 0; $i < $appliance['duration']; $i++) {
                        $currentHour = $startHour + $i;
                        $schedule[$day][$currentHour] = [
                            'appliance_id' => $appliance['id'],
                            'day' => $day,
                            'start_hour' => $currentHour,
                            'end_hour' => $currentHour + 1, // Assuming 1-hour slots
                        ];
                    }
                }
            }
        }

        // Log the schedule for debugging
        \Log::info('Generated schedule:', ['schedule' => $schedule]);

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
            'preferred_start' => 'required|date_format:H:i',
            'preferred_end' => 'required|date_format:H:i',
            'duration' => 'required|numeric',
        ]);

        Appliance::create([
            'name' => $request->input('name'),
            'power' => $request->input('power'),
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
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
        \Log::info('Request Data:', $request->all());

        try {
            $validatedData = $request->validate([
                'name' => 'required|string',
                'power' => 'required|numeric',
                'preferred_start' => 'required|date_format:H:i',
                'preferred_end' => 'required|date_format:H:i',
                'duration' => 'required|numeric',
            ]);

            \Log::info('Validation Passed:', $validatedData);

            $appliance = Appliance::find($id);

            if (!$appliance) {
                \Log::error('Appliance not found:', ['id' => $id]);
                return redirect()->route('appliances.manage')->with('error', 'Appliance not found.');
            }

            \Log::info('Appliance ID:', ['id' => $id]);
            \Log::info('Appliance Data Before Update:', $appliance->toArray());

            $updated = $appliance->update($validatedData);

            \Log::info('Update Result:', ['updated' => $updated]);
            \Log::info('Updated Appliance:', $appliance->toArray());

            return redirect()->route('appliances.manage')->with('success', 'Appliance updated successfully.');
        } catch (\Illuminate\Validation\ValidationException $e) {
            \Log::error('Validation Failed:', $e->errors());
            return redirect()->back()->withErrors($e->errors())->withInput();
        }
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

    public function addSelectedAppliance(Request $request)
    {
        $request->validate([
            'appliance_id' => 'required|integer',
            'name' => 'required|string',
            'preferred_start' => 'required|date_format:H:i',
            'preferred_end' => 'required|date_format:H:i',
            'duration' => 'required|numeric',
            'usage_days' => 'required|array',
        ]);

        // Update the appliance in the appliances table
        $appliance = Appliance::find($request->input('appliance_id'));
        if ($appliance) {
            $appliance->update([
                'preferred_start' => $request->input('preferred_start'),
                'preferred_end' => $request->input('preferred_end'),
                'duration' => $request->input('duration'),
            ]);
        }

        // Save the selected appliance to the selected_appliances table
        SelectedAppliance::create([
            'appliance_id' => $request->input('appliance_id'),
            'name' => $request->input('name'),
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
            'usage_days' => json_encode($request->input('usage_days')), // Save as JSON
        ]);

        return response()->json(['success' => true]);
    }

    public function createSchedule()
    {
        $appliances = Appliance::all();
        $selectedAppliances = SelectedAppliance::all();

        return view('schedule', [
            'appliances' => $appliances,
            'selectedAppliances' => $selectedAppliances,
        ]);
    }
}