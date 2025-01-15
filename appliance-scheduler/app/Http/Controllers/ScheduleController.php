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
        // Fetch predictions from the session
        $predictions = session('predictions', []);

        // Clear predictions from the session (optional)
        session()->forget('predictions');

        // Fetch appliances and selected appliances with the appliance relationship
        $appliances = Appliance::all();
        $selectedAppliances = SelectedAppliance::with('appliance')->get();

        // Fetch the schedule from the `schedules` table
        $schedule = Schedule::with('appliance')->get();

        // Debugging: Check for null appliance relationships
        foreach ($selectedAppliances as $selectedAppliance) {
            if (!$selectedAppliance->appliance) {
                \Log::warning('SelectedAppliance has no associated Appliance', [
                    'selected_appliance_id' => $selectedAppliance->id,
                    'appliance_id' => $selectedAppliance->appliance_id,
                ]);
            }
        }

        // Pass the data to the view
        return view('dashboard', [
            'appliances' => $appliances,
            'selectedAppliances' => $selectedAppliances,
            'schedule' => $schedule,
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

        // Clear the schedules table before generating a new schedule
        Schedule::truncate();

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

            // Store predictions in the session
            session(['predictions' => $predictions]);

            // Retrieve selected appliances from the database
            $appliances = SelectedAppliance::all()->map(function ($selectedAppliance) {
                return [
                    'id' => $selectedAppliance->appliance_id, // Use appliance_id instead of id
                    'name' => $selectedAppliance->name,
                    'power' => $selectedAppliance->power,
                    'preferred_start' => $selectedAppliance->preferred_start,
                    'preferred_end' => $selectedAppliance->preferred_end,
                    'duration' => $selectedAppliance->duration,
                    'usage_days' => $selectedAppliance->usage_days,
                ];
            })->toArray();

            // Schedule the appliances
            $this->scheduleAppliances($predictions, $appliances);

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

        \Log::info('Predictions:', $predictions);
        \Log::info('Appliances:', $appliances);

        // Sort predictions by price (ascending)
        usort($predictions, function ($a, $b) {
            return (float) $a['Predicted Price [Euro/MWh]'] <=> (float) $b['Predicted Price [Euro/MWh]'];
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
            $priorityA = (float) $a['power'] * (float) $a['duration'];
            $priorityB = (float) $b['power'] * (float) $b['duration'];
            return $priorityB <=> $priorityA; // Sort in descending order
        });

        // Assign appliances to the cheapest hours within their preferred time slots
        foreach ($appliances as $appliance) {
            $day = $appliance['usage_days']; // usage_days is now a string (e.g., "Monday")

            if (!isset($predictionsByDay[$day])) {
                \Log::warning('No predictions for day:', ['day' => $day]);
                continue; // Skip if no predictions for this day
            }

            // Log the appliance and its preferences
            \Log::info('Scheduling appliance:', [
                'appliance_id' => $appliance['id'],
                'day' => $day,
                'preferred_start' => $appliance['preferred_start'],
                'preferred_end' => $appliance['preferred_end'],
                'duration' => $appliance['duration'],
            ]);

            // Find all possible windows within the preferred time range
            $windows = [];
            for ($startHour = (int) $appliance['preferred_start']; $startHour + (int) $appliance['duration'] - 1 <= (int) $appliance['preferred_end']; $startHour++) {
                // Calculate the total cost for this window
                $totalCost = 0;
                $conflict = false;

                for ($i = 0; $i < (int) $appliance['duration']; $i++) {
                    $currentHour = $startHour + $i;

                    // Check if the hour is available
                    if (isset($schedule[$day][$currentHour])) {
                        $conflict = true;
                        break;
                    }

                    // Add the price for this hour
                    if (isset($predictionsByDay[$day][$currentHour])) {
                        $totalCost += (float) $predictionsByDay[$day][$currentHour]['Predicted Price [Euro/MWh]'];
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
            $assigned = false;
            foreach ($windows as $window) {
                $startHour = $window['startHour'];

                // Check if the window is still available
                $conflict = false;
                for ($i = 0; $i < (int) $appliance['duration']; $i++) {
                    if (isset($schedule[$day][$startHour + $i])) {
                        $conflict = true;
                        break;
                    }
                }

                if (!$conflict) {
                    // Calculate predicted start and end times
                    $predictedStartTime = date('H:i', strtotime("$startHour:00"));
                    $predictedEndTime = date('H:i', strtotime("$startHour:00") + ($appliance['duration'] * 3600));

                    // Log the data being inserted
                    \Log::info('Creating schedule record:', [
                        'appliance_id' => $appliance['id'],
                        'day' => $day,
                        'start_hour' => $startHour,
                        'end_hour' => $startHour + $appliance['duration'],
                        'predicted_start_time' => $predictedStartTime,
                        'predicted_end_time' => $predictedEndTime,
                    ]);

                    // Save the schedule to the `schedules` table
                    Schedule::create([
                        'appliance_id' => $appliance['id'],
                        'day' => $day,
                        'start_hour' => $startHour,
                        'end_hour' => $startHour + $appliance['duration'],
                        'predicted_start_time' => $predictedStartTime,
                        'predicted_end_time' => $predictedEndTime,
                        'created_at' => now(),
                        'updated_at' => now(),
                    ]);

                    // Mark the time slot as occupied
                    for ($i = 0; $i < (int) $appliance['duration']; $i++) {
                        $schedule[$day][$startHour + $i] = true;
                    }

                    $assigned = true;
                    break; // Exit the loop after assigning the appliance
                }
            }

            // If no window is available within the preferred range, try to assign outside the preferred range
            if (!$assigned) {
                \Log::info('No available windows within preferred range for appliance:', [
                    'appliance_id' => $appliance['id'],
                    'day' => $day,
                ]);

                // Try to assign the appliance to the cheapest available window outside the preferred range
                $allWindows = [];
                for ($startHour = 0; $startHour + (int) $appliance['duration'] - 1 <= 23; $startHour++) {
                    // Calculate the total cost for this window
                    $totalCost = 0;
                    $conflict = false;

                    for ($i = 0; $i < (int) $appliance['duration']; $i++) {
                        $currentHour = $startHour + $i;

                        // Check if the hour is available
                        if (isset($schedule[$day][$currentHour])) {
                            $conflict = true;
                            break;
                        }

                        // Add the price for this hour
                        if (isset($predictionsByDay[$day][$currentHour])) {
                            $totalCost += (float) $predictionsByDay[$day][$currentHour]['Predicted Price [Euro/MWh]'];
                        } else {
                            $conflict = true; // Skip if any hour in the window is missing
                            break;
                        }
                    }

                    if (!$conflict) {
                        $allWindows[] = [
                            'startHour' => $startHour,
                            'totalCost' => $totalCost,
                        ];
                    }
                }

                // Sort all windows by total cost (ascending)
                usort($allWindows, function ($a, $b) {
                    return $a['totalCost'] <=> $b['totalCost'];
                });

                // Assign the appliance to the cheapest available window
                if (!empty($allWindows)) {
                    $cheapestWindow = $allWindows[0];
                    $startHour = $cheapestWindow['startHour'];

                    // Calculate predicted start and end times
                    $predictedStartTime = date('H:i', strtotime("$startHour:00"));
                    $predictedEndTime = date('H:i', strtotime("$startHour:00") + ($appliance['duration'] * 3600));

                    // Log the data being inserted
                    \Log::info('Creating schedule record outside preferred range:', [
                        'appliance_id' => $appliance['id'],
                        'day' => $day,
                        'start_hour' => $startHour,
                        'end_hour' => $startHour + $appliance['duration'],
                        'predicted_start_time' => $predictedStartTime,
                        'predicted_end_time' => $predictedEndTime,
                    ]);

                    // Save the schedule to the `schedules` table
                    Schedule::create([
                        'appliance_id' => $appliance['id'],
                        'day' => $day,
                        'start_hour' => $startHour,
                        'end_hour' => $startHour + $appliance['duration'],
                        'predicted_start_time' => $predictedStartTime,
                        'predicted_end_time' => $predictedEndTime,
                        'created_at' => now(),
                        'updated_at' => now(),
                    ]);

                    // Mark the time slot as occupied
                    for ($i = 0; $i < (int) $appliance['duration']; $i++) {
                        $schedule[$day][$startHour + $i] = true;
                    }
                } else {
                    \Log::warning('No available windows for appliance:', [
                        'appliance_id' => $appliance['id'],
                        'day' => $day,
                    ]);
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
            // Validate the request data
            $validatedData = $request->validate([
                'name' => 'required|string',
                'power' => 'required|numeric',
                'preferred_start' => 'required|date_format:H:i', // Ensure time is in H:i format
                'preferred_end' => 'required|date_format:H:i',   // Ensure time is in H:i format
                'duration' => 'required|numeric',
            ]);

            \Log::info('Validation Passed:', $validatedData);

            // Find the appliance
            $appliance = Appliance::find($id);

            if (!$appliance) {
                \Log::error('Appliance not found:', ['id' => $id]);
                return redirect()->route('appliances.manage')->with('error', 'Appliance not found.');
            }

            \Log::info('Appliance ID:', ['id' => $id]);
            \Log::info('Appliance Data Before Update:', $appliance->toArray());

            // Update the appliance with validated data
            $appliance->update([
                'name' => $validatedData['name'],
                'power' => $validatedData['power'],
                'preferred_start' => $validatedData['preferred_start'], // Ensure time is in H:i format
                'preferred_end' => $validatedData['preferred_end'],     // Ensure time is in H:i format
                'duration' => $validatedData['duration'],
            ]);

            \Log::info('Update Result:', ['updated' => true]);
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
            'preferred_start' => 'required|date_format:H:i',
            'preferred_end' => 'required|date_format:H:i',
            'duration' => 'required|numeric',
            'usage_days' => 'required|string|in:Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday',
        ]);

        // Retrieve the appliance from the appliance table
        $appliance = Appliance::find($request->input('appliance_id'));

        if (!$appliance) {
            return response()->json(['success' => false, 'message' => 'Appliance not found.'], 404);
        }

        // Update the appliance in the appliances table
        $appliance->update([
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
        ]);

        // Save the selected appliance to the selected_appliance table
        SelectedAppliance::create([
            'appliance_id' => $request->input('appliance_id'),
            'name' => $appliance->name, // Get name from appliance table
            'power' => $appliance->power, // Get power from appliance table
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
            'usage_days' => $request->input('usage_days'), // Save as a string
        ]);

        return response()->json(['success' => true]);
    }

    public function removeSelectedAppliance($id)
    {
        try {
            // Find the selected appliance
            $selectedAppliance = SelectedAppliance::find($id);

            if (!$selectedAppliance) {
                return response()->json(['success' => false, 'message' => 'Appliance not found.'], 404);
            }

            // Delete the selected appliance
            $selectedAppliance->delete();

            return response()->json(['success' => true]);
        } catch (\Exception $e) {
            \Log::error('Error removing appliance:', ['error' => $e->getMessage()]);
            return response()->json(['success' => false, 'message' => 'An error occurred. Please try again.'], 500);
        }
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