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

        // Initialize schedule tracking appliance counts per hour
        $schedule = [];
        $penalty = 0.1; // Adjustable penalty factor (10% per appliance)

        // Group predictions by day
        $predictionsByDay = [];
        foreach ($predictions as $prediction) {
            $day = date('l', strtotime($prediction['Start date/time']));
            $hour = (int) date('H', strtotime($prediction['Start date/time']));
            $predictionsByDay[$day][$hour] = $prediction;
        }

        // Sort appliances by priority (power * duration)
        usort($appliances, function ($a, $b) {
            $priorityA = (float) $a['power'] * (float) $a['duration'];
            $priorityB = (float) $b['power'] * (float) $b['duration'];
            return $priorityB <=> $priorityA;
        });

        foreach ($appliances as $appliance) {
            $day = $appliance['usage_days'];
            if (!isset($predictionsByDay[$day])) {
                \Log::warning('No predictions for day:', ['day' => $day]);
                continue;
            }

            $duration = (int) $appliance['duration'];
            $preferredStart = (int) $appliance['preferred_start'];
            $preferredEnd = (int) $appliance['preferred_end'];

            \Log::info('Scheduling appliance:', [
                'appliance_id' => $appliance['id'],
                'day' => $day,
                'preferred_start' => $preferredStart,
                'preferred_end' => $preferredEnd,
                'duration' => $duration,
            ]);

            // First try: Preferred time window
            $windows = $this->findWindows(
                $predictionsByDay[$day],
                $schedule[$day] ?? [],
                $preferredStart,
                $preferredEnd,
                $duration,
                $penalty
            );

            // Fallback: Entire day if no preferred window found
            if (empty($windows)) {
                \Log::info('No preferred windows found, searching full day', [
                    'appliance_id' => $appliance['id']
                ]);

                $windows = $this->findWindows(
                    $predictionsByDay[$day],
                    $schedule[$day] ?? [],
                    0, // Start at midnight
                    23 - $duration + 1, // Allow full day search
                    $duration,
                    $penalty
                );
            }

            if (!empty($windows)) {
                usort($windows, function ($a, $b) {
                    return $a['adjustedCost'] <=> $b['adjustedCost'];
                });
                $this->assignWindow($windows[0], $appliance, $day, $schedule);
            } else {
                \Log::error('CRITICAL: Failed to schedule appliance', [
                    'appliance_id' => $appliance['id'],
                    'day' => $day
                ]);
                throw new \Exception("Failed to schedule appliance {$appliance['id']}");
            }
        }

        \Log::info('Generated schedule:', ['schedule' => $schedule]);
        return $schedule;
    }

    private function findWindows($dayPredictions, $daySchedule, $startMin, $endMax, $duration, $penalty)
    {
        $windows = [];

        for ($startHour = $startMin; $startHour <= $endMax; $startHour++) {
            $adjustedCost = 0;
            $validWindow = true;

            for ($i = 0; $i < $duration; $i++) {
                $currentHour = $startHour + $i;

                if (!isset($dayPredictions[$currentHour])) {
                    $validWindow = false;
                    break;
                }

                $basePrice = (float) $dayPredictions[$currentHour]['Predicted Price [Euro/MWh]'];
                $applianceCount = $daySchedule[$currentHour] ?? 0;
                $adjustedCost += $basePrice * (1 + $penalty * $applianceCount);
            }

            if ($validWindow) {
                $windows[] = [
                    'startHour' => $startHour,
                    'adjustedCost' => $adjustedCost,
                    'inPreferred' => ($startHour >= $startMin &&
                        ($startHour + $duration - 1) <= $endMax)
                ];
            }
        }

        return $windows;
    }

    private function assignWindow($window, $appliance, $day, &$schedule)
    {
        $startHour = $window['startHour'];
        $duration = (int) $appliance['duration'];

        // Create schedule record
        Schedule::create([
            'appliance_id' => $appliance['id'],
            'day' => $day,
            'start_hour' => $startHour,
            'end_hour' => $startHour + $duration,
            'predicted_start_time' => date('H:i', strtotime("$startHour:00")),
            'predicted_end_time' => date('H:i', strtotime("$startHour:00") + ($duration * 3600)),
            'created_at' => now(),
            'updated_at' => now(),
            'within_preferred' => $window['inPreferred'] ?? false
        ]);

        // Update appliance counts in schedule
        for ($i = 0; $i < $duration; $i++) {
            $currentHour = $startHour + $i;
            $schedule[$day][$currentHour] = ($schedule[$day][$currentHour] ?? 0) + 1;
        }

        \Log::info('Assigned appliance ' . ($window['inPreferred'] ? 'within' : 'outside') . ' preferred time', [
            'appliance_id' => $appliance['id'],
            'start_hour' => $startHour
        ]);
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