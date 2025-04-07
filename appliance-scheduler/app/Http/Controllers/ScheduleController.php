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
        // Fetch predictions from the session (if any) and do NOT forget them immediately
        // so that we can use them for cost calculations.
        $predictions = session('predictions', []);

        // Build an associative array: $prices[day][hour] = predicted price (€/MWh)
        $pricesByDayHour = [];
        foreach ($predictions as $prediction) {
            $dt = \Carbon\Carbon::parse($prediction['StartDateTime']);
            $day = $dt->format('l'); // e.g., "Monday"
            $hour = (int) $dt->format('H');
            $pricesByDayHour[$day][$hour] = $prediction['Predicted_Price'];
        }

        // Fetch appliances and schedules as before
        $appliances = Appliance::all();
        $selectedAppliances = SelectedAppliance::with('appliance')->get();
        $schedule = Schedule::with('appliance')->get();

        $scheduledLoadByDayHour = [];

        // Calculate weekly cost by summing, for each schedule entry, over its hours:
        $weeklyCost = 0;

        foreach ($schedule as $s) {
            $duration = $s->end_hour - $s->start_hour;
            $power = $s->appliance ? $s->appliance->power : 0;
            for ($h = $s->start_hour; $h < $s->end_hour; $h++) {
                // Initialize day/hour if not set
                if (!isset($scheduledLoadByDayHour[$s->day])) {
                    $scheduledLoadByDayHour[$s->day] = [];
                }
                $scheduledLoadByDayHour[$s->day][$h] = ($scheduledLoadByDayHour[$s->day][$h] ?? 0) + $power;
            }

            $costForSchedule = 0;
            for ($h = $s->start_hour; $h < $s->end_hour; $h++) {
                if (isset($pricesByDayHour[$s->day][$h])) {
                    // Convert €/MWh to €/kWh by dividing by 1000
                    $pricePerKWh = $pricesByDayHour[$s->day][$h] / 1000;
                    // Cost for this hour: power (kW) * pricePerKWh (€/kWh)
                    $costForSchedule += $power * $pricePerKWh;
                }
            }
            $weeklyCost += $costForSchedule;
        }

        $baselineSchedule = $this->getBaselineSchedule($selectedAppliances->toArray());
        $baselineLoadByDayHour = [];
        $baselineCost = 0;
        foreach ($baselineSchedule as $s) {
            $duration = $s['end_hour'] - $s['start_hour'];
            $power = $s['power'];

            for ($h = $s['start_hour']; $h < $s['end_hour']; $h++) {
                // Track baseline load
                $baselineLoadByDayHour[$s['day']][$h] = ($baselineLoadByDayHour[$s['day']][$h] ?? 0) + $power;

                // Track baseline cost
                if (isset($pricesByDayHour[$s['day']][$h])) {
                    $pricePerKWh = $pricesByDayHour[$s['day']][$h] / 1000;
                    $baselineCost += $power * $pricePerKWh;
                }
            }
        }

        $baselinePAR = $this->calculatePAR($baselineLoadByDayHour);
        \Log::info("Baseline Loads:", $baselineLoadByDayHour);
        \Log::info("Scheduled Load Data:", $scheduledLoadByDayHour);
        $scheduledPAR = $this->calculatePAR($scheduledLoadByDayHour);

        // Calculate cost reduction
        $costReduction = ($baselineCost - $weeklyCost) / $baselineCost * 100;

        // Print results to terminal
        \Log::info("RESULTS:");
        \Log::info("Baseline PAR: " . number_format($baselinePAR, 2));
        \Log::info("Scheduled PAR: " . number_format($scheduledPAR, 2));
        \Log::info("Cost Reduction: " . number_format($costReduction, 2) . "%");


        return view('dashboard', [
            'appliances' => $appliances,
            'selectedAppliances' => $selectedAppliances,
            'schedule' => $schedule,
            'predictions' => $predictions,
            'weeklyCost' => $weeklyCost,
        ]);
    }

    private function calculatePAR($loadByDayHour)
    {
        $allLoads = [];
        foreach ($loadByDayHour as $day => $hours) {
            foreach ($hours as $hour => $load) {
                if ($load > 0)
                    $allLoads[] = $load; // Skip zero loads
            }
        }

        if (empty($allLoads))
            return 0; // Or return NAN to flag errors

        $peak = max($allLoads);
        $average = array_sum($allLoads) / count($allLoads);
        return $peak / $average;
    }

    private function getBaselineSchedule($appliances)
    {
        // Define custom start times for each appliance (24-hour format)
        $applianceStartTimes = [
            'Oven' => 18,      // 6 PM (evening cooking)
            'Dishwasher' => 12, // 12 PM (noon)
            'Washing Machine' => 9, // 9 AM
            'Electric Heater' => 1, // 12 AM (runs overnight)
            'Vacuum' => 15      // 3 PM
        ];

        $baselineSchedule = [];

        foreach ($appliances as $appliance) {
            $name = $appliance['name'];
            $startHour = $applianceStartTimes[$name] ?? 8; // Default to 8 AM if not found

            $baselineSchedule[] = [
                'day' => $appliance['usage_days'], // Preserve original usage day
                'start_hour' => $startHour,
                'end_hour' => $startHour + $appliance['duration'],
                'power' => $appliance['power'],
                'name' => $name
            ];
        }

        return $baselineSchedule;
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
            $response = Http::timeout(300)->post('http://127.0.0.1:8001/predict', [
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

        // Sort predictions by price 
        usort($predictions, function ($a, $b) {
            return (float) $a['Predicted_Price'] <=> (float) $b['Predicted_Price'];
        });

        // Initialize schedule tracking appliance counts per hour
        $schedule = [];

        // Group predictions by day
        $predictionsByDay = [];
        foreach ($predictions as $prediction) {
            $day = date('l', strtotime($prediction['StartDateTime']));
            $hour = (int) date('H', strtotime($prediction['StartDateTime']));
            $predictionsByDay[$day][$hour] = $prediction;
        }


        foreach ($predictionsByDay as $day => &$hours) {
            ksort($hours);
            $prices = array_column($hours, 'Predicted_Price');

            // Dynamic thresholds
            $medianPrice = $this->calculatePercentile($prices, 50);
            $peakThreshold = $this->calculatePercentile($prices, 75);

            foreach ($hours as $hour => $data) {
                $price = (float) $data['Predicted_Price'];
                $hours[$hour]['isPeak'] = ($price >= $peakThreshold);
                $hours[$hour]['peakPenalty'] = max(0, ($price - $medianPrice) / $medianPrice);
            }
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
                    $duration
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

    private function findWindows($dayPredictions, $daySchedule, $startMin, $endMax, $duration)
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

                $basePrice = (float) $dayPredictions[$currentHour]['Predicted_Price'];
                $applianceCount = $daySchedule[$currentHour] ?? 0;
                $peakPenalty = $dayPredictions[$currentHour]['peakPenalty'] ?? 0;

                // Calibrated penalties
                $congestionPenalty = 0.03 * pow($applianceCount, 2);
                $adjustedCost += $basePrice * (1 + $congestionPenalty + $peakPenalty);
            }

            if ($validWindow) {
                $windows[] = [
                    'startHour' => $startHour,
                    'adjustedCost' => $adjustedCost,
                    'inPreferred' => ($startHour >= $startMin && ($startHour + $duration - 1) <= $endMax)
                ];
            }
        }
        return $windows;
    }

    private function calculatePercentile($array, $percentile)
    {
        sort($array);
        $index = ($percentile / 100) * (count($array) - 1);
        $floor = floor($index);
        $fraction = $index - $floor;
        return $floor + 1 < count($array)
            ? $array[$floor] + $fraction * ($array[$floor + 1] - $array[$floor])
            : $array[$floor];
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
        \Log::info('updateAppliance() triggered');
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
        $selectedAppliance = SelectedAppliance::create([
            'appliance_id' => $request->input('appliance_id'),
            'name' => $appliance->name,
            'power' => $appliance->power,
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
            'usage_days' => $request->input('usage_days'),
        ]);

        // Return the ID of the new selected appliance
        return response()->json([
            'success' => true,
            'selected_appliance_id' => $selectedAppliance->id,
        ]);
    }

    public function getSelectedAppliance($id)
    {
        $appliance = SelectedAppliance::find($id);
        if (!$appliance) {
            return response()->json(['success' => false, 'message' => 'Appliance not found'], 404);
        }
        return response()->json(['success' => true, 'appliance' => $appliance]);
    }

    public function updateSelectedAppliance(Request $request, $id)
    {
        \Log::info('Update request received:', $request->all());

        $request->validate([
            'preferred_start' => 'required|date_format:H:i',
            'preferred_end' => 'required|date_format:H:i',
            'duration' => 'required|numeric|min:0.01',
        ]);

        $appliance = SelectedAppliance::find($id);
        if (!$appliance) {
            return response()->json(['success' => false, 'message' => 'Appliance not found'], 404);
        }

        $appliance->update([
            'preferred_start' => $request->preferred_start,
            'preferred_end' => $request->preferred_end,
            'duration' => $request->duration,
        ]);

        \Log::info('Appliance updated successfully', $appliance->toArray());

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