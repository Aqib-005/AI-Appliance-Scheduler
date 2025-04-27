<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use App\Models\Appliance;
use App\Models\SelectedAppliance;
use App\Models\Schedule;

class ScheduleController
{
    public function dashboard()
    {
        //  Get cached predictions if there is any
        $predictions = session('predictions', []);

        // Convert predictions into price array
        $pricesByDayHour = [];
        foreach ($predictions as $prediction) {
            $dt = \Carbon\Carbon::parse($prediction['StartDateTime']);
            $day = $dt->format('l');
            $hour = (int) $dt->format('H');
            $pricesByDayHour[$day][$hour] = $prediction['Predicted_Price'];
        }

        // Load data from database
        $appliances = Appliance::all();
        $selectedAppliances = SelectedAppliance::with('appliance')->get();
        $schedule = Schedule::with('appliance')->get();

        // Calculate actual scheduled load and weekly cost
        $scheduledLoadByDayHour = [];
        $weeklyCost = 0;
        foreach ($schedule as $s) {
            $duration = $s->end_hour - $s->start_hour;
            $power = $s->appliance ? $s->appliance->power : 0;

            // Accumulate kW load per hour
            for ($h = $s->start_hour; $h < $s->end_hour; $h++) {
                $scheduledLoadByDayHour[$s->day][$h] =
                    ($scheduledLoadByDayHour[$s->day][$h] ?? 0) + $power;
            }

            // Sum cost using price lookup
            $costForSchedule = 0;
            for ($h = $s->start_hour; $h < $s->end_hour; $h++) {
                if (isset($pricesByDayHour[$s->day][$h])) {
                    $pricePerKWh = $pricesByDayHour[$s->day][$h] / 1000;
                    $costForSchedule += $power * $pricePerKWh;
                }
            }
            $weeklyCost += $costForSchedule;
        }

        // Simple baseline schedule for comparison (not useed for actual logic of webapp)
        $baselineSchedule = $this->getBaselineSchedule($selectedAppliances->toArray());
        $baselineLoadByDayHour = [];
        $baselineCost = 0;
        foreach ($baselineSchedule as $s) {
            $duration = $s['end_hour'] - $s['start_hour'];
            $power = $s['power'];

            for ($h = $s['start_hour']; $h < $s['end_hour']; $h++) {
                $baselineLoadByDayHour[$s['day']][$h] =
                    ($baselineLoadByDayHour[$s['day']][$h] ?? 0) + $power;
                if (isset($pricesByDayHour[$s['day']][$h])) {
                    $baselineCost += $power * ($pricesByDayHour[$s['day']][$h] / 1000);
                }
            }
        }

        // Compute Peak-to-Average Ratio 
        $baselinePAR = $this->calculatePAR($baselineLoadByDayHour);
        $scheduledPAR = $this->calculatePAR($scheduledLoadByDayHour);
        \Log::info("Baseline PAR: " . number_format($baselinePAR, 2));
        \Log::info("Scheduled PAR: " . number_format($scheduledPAR, 2));

        // Render the dashboard view 
        return view('dashboard', [
            'appliances' => $appliances,
            'selectedAppliances' => $selectedAppliances,
            'schedule' => $schedule,
            'predictions' => $predictions,
            'weeklyCost' => $weeklyCost,
        ]);
    }

    // Compute PAR for dayhour load array
    private function calculatePAR($loadByDayHour)
    {
        $allLoads = [];
        // Flatten all positive loads
        foreach ($loadByDayHour as $hours) {
            foreach ($hours as $load) {
                if ($load > 0) {
                    $allLoads[] = $load;
                }
            }
        }

        //  PAR is zero if empty 
        if (empty($allLoads)) {
            return 0;
        }

        $peak = max($allLoads);
        $average = array_sum($allLoads) / count($allLoads);
        return $peak / $average;
    }

    // Baseline hardcoded schedule (not relevant and only used for benchmarking results)
    private function getBaselineSchedule($appliances)
    {
        $applianceStartTimes = [
            'Oven' => 6,
            'Dishwasher' => 12,
            'Washing Machine' => 14,
            'Electric Heater' => 15,
            'Vacuum' => 15,
        ];

        $baselineSchedule = [];
        foreach ($appliances as $appliance) {
            $name = $appliance['name'];
            $startHour = $applianceStartTimes[$name] ?? 8;
            $baselineSchedule[] = [
                'day' => $appliance['usage_days'],
                'start_hour' => $startHour,
                'end_hour' => $startHour + $appliance['duration'],
                'power' => $appliance['power'],
                'name' => $name,
            ];
        }
        return $baselineSchedule;
    }

    // Display results
    public function showResults()
    {
        return view('results', [
            'schedule' => session('schedule'),
            'predictions' => session('predictions'),
        ]);
    }

    // Generate a new schedule
    public function store(Request $request)
    {
        set_time_limit(300);

        // Clear  previous schedule
        Schedule::truncate();

        // Determine week-start date (this Monday)
        $start_date = date('Y-m-d', strtotime('monday this week'));
        \Log::info('Calculated start_date:', ['start_date' => $start_date]);

        try {
            // Call ML API
            $start_time_ml = microtime(true);
            $response = Http::timeout(300)->post('http://127.0.0.1:8001/predict', [
                'start_date' => $start_date,
            ]);
            $runtime_ml = microtime(true) - $start_time_ml;
            \Log::info("ML prediction runtime: " . number_format($runtime_ml, 2) . " seconds");

            \Log::info('FastAPI response', [
                'status' => $response->status(),
                'body' => $response->body(),
            ]);
            if ($response->failed()) {
                throw new \Exception('Failed to fetch predictions');
            }

            // Cache predictions
            $predictions = $response->json()['predictions'];
            session(['predictions' => $predictions]);

            // Build array of selected appliances
            $appliances = SelectedAppliance::all()
                ->map(fn($sa) => [
                    'id' => $sa->appliance_id,
                    'name' => $sa->name,
                    'power' => $sa->power,
                    'preferred_start' => $sa->preferred_start,
                    'preferred_end' => $sa->preferred_end,
                    'duration' => $sa->duration,
                    'usage_days' => $sa->usage_days,
                ])
                ->toArray();

            // Run scheduling algorithm 
            $start_time_sched = microtime(true);
            $this->scheduleAppliances($predictions, $appliances);
            $runtime_sched = microtime(true) - $start_time_sched;
            \Log::info("Scheduling algorithm runtime: " . number_format($runtime_sched, 2) . " seconds");

            return response()->json([
                'success' => true,
                'redirect_url' => route('dashboard'),
            ]);
        } catch (\Exception $e) {
            \Log::error('Error in store method:', ['error' => $e->getMessage()]);
            return response()->json([
                'success' => false,
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    // Scheduling algorithm 
    private function scheduleAppliances($predictions, $appliances)
    {
        set_time_limit(300);
        \Log::info('Predictions:', $predictions);
        \Log::info('Appliances:', $appliances);

        // Sort predictions ascending by price
        usort(
            $predictions,
            fn($a, $b) =>
            (float) $a['Predicted_Price'] <=> (float) $b['Predicted_Price']
        );

        // Group predictions by day and hour
        $predictionsByDay = [];
        foreach ($predictions as $p) {
            $day = date('l', strtotime($p['StartDateTime']));
            $hour = (int) date('H', strtotime($p['StartDateTime']));
            $predictionsByDay[$day][$hour] = $p;
        }

        // Compute median and peak threshold
        foreach ($predictionsByDay as $day => &$hours) {
            ksort($hours);
            $prices = array_column($hours, 'Predicted_Price');
            $medianPrice = $this->calculatePercentile($prices, 50);
            $peakThreshold = $this->calculatePercentile($prices, 75);
            foreach ($hours as $h => $row) {
                $price = (float) $row['Predicted_Price'];
                $hours[$h]['isPeak'] = $price >= $peakThreshold;
                $hours[$h]['peakPenalty'] = max(0, ($price - $medianPrice) / $medianPrice);
            }
        }

        // Order appliances by (power × duration) descending
        usort(
            $appliances,
            fn($a, $b) =>
            ($b['power'] * $b['duration']) <=> ($a['power'] * $a['duration'])
        );

        // Attempt to schedule each appliance
        $schedule = [];
        foreach ($appliances as $app) {
            $day = $app['usage_days'];
            $duration = (int) $app['duration'];
            $preferredStart = (int) $app['preferred_start'];
            $preferredEnd = (int) $app['preferred_end'];

            // Find windows within preferred block
            $windows = $this->findWindows(
                $predictionsByDay[$day] ?? [],
                $schedule[$day] ?? [],
                $preferredStart,
                $preferredEnd - $duration,
                $duration
            );

            // Fallback if none found
            if (empty($windows)) {
                $windows = $this->findWindows(
                    $predictionsByDay[$day] ?? [],
                    $schedule[$day] ?? [],
                    0,
                    24 - $duration,
                    $duration
                );
            }

            // If still empty throw error
            if (empty($windows)) {
                \Log::error("No valid window for appliance {$app['id']} on $day");
                throw new \Exception("Failed to schedule appliance {$app['id']}");
            }

            // Choose cheapest window
            usort($windows, fn($a, $b) => $a['adjustedCost'] <=> $b['adjustedCost']);
            $this->assignWindow($windows[0], $app, $day, $schedule);
        }

        \Log::info('Generated schedule:', ['schedule' => $schedule]);
        return $schedule;
    }

    // Helper fucntion to find valid windows
    private function findWindows($dayPred, $daySched, $startMin, $endMax, $duration)
    {
        $windows = [];
        for ($h = $startMin; $h <= $endMax; $h++) {
            $valid = true;
            $adjustedCost = 0;
            for ($i = 0; $i < $duration; $i++) {
                $hour = $h + $i;
                if (!isset($dayPred[$hour])) {
                    $valid = false;
                    break;
                }
                $basePrice = (float) $dayPred[$hour]['Predicted_Price'];
                $count = $daySched[$hour] ?? 0;
                $peakPenalty = $dayPred[$hour]['peakPenalty'] ?? 0;
                $congPenalty = 0.03 * $count * $count;
                $adjustedCost += $basePrice * (1 + $peakPenalty + $congPenalty);
            }
            if ($valid) {
                $windows[] = [
                    'startHour' => $h,
                    'adjustedCost' => $adjustedCost,
                    'inPreferred' => ($h >= $startMin && ($h + $duration - 1) <= $endMax),
                ];
            }
        }
        return $windows;
    }

    // Percentile calculation
    private function calculatePercentile($array, $percentile)
    {
        sort($array);
        $index = ($percentile / 100) * (count($array) - 1);
        $floor = floor($index);
        $fraction = $index - $floor;
        if ($floor + 1 < count($array)) {
            return $array[$floor] + $fraction * ($array[$floor + 1] - $array[$floor]);
        }
        return $array[$floor];
    }

    // Assign window and apply penalty
    private function assignWindow($window, $appliance, $day, &$schedule)
    {
        $startHour = $window['startHour'];
        $duration = (int) $appliance['duration'];

        // Create the Schedule record
        Schedule::create([
            'appliance_id' => $appliance['id'],
            'day' => $day,
            'start_hour' => $startHour,
            'end_hour' => $startHour + $duration,
            'predicted_start_time' => date('H:i', strtotime("$startHour:00")),
            'predicted_end_time' => date('H:i', strtotime("$startHour:00") + $duration * 3600),
            'within_preferred' => $window['inPreferred'] ?? false,
        ]);

        // Update congestion penalties
        for ($i = 0; $i < $duration; $i++) {
            $hour = $startHour + $i;
            $schedule[$day][$hour] = ($schedule[$day][$hour] ?? 0) + 1;
        }
    }

    // Get Appliance details as JSON
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

    // Form submission to add a new appliance
    public function addAppliance(Request $request)
    {
        // Validate inputs
        $request->validate([
            'name' => 'required|string',
            'power' => 'required|numeric',
            'preferred_start' => 'required|date_format:H:i',
            'preferred_end' => 'required|date_format:H:i',
            'duration' => 'required|numeric',
        ]);

        // Create new appliance record
        Appliance::create([
            'name' => $request->input('name'),
            'power' => $request->input('power'),
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
        ]);

        return redirect()->route('appliances.manage')
            ->with('success', 'Appliance added successfully.');
    }

    // Delete an appliance by ID
    public function removeAppliance($id)
    {
        $appliance = Appliance::find($id);
        if ($appliance) {
            $appliance->delete();
            return response()->json(['success' => true]);
        }
        return response()->json(['success' => false], 400);
    }

    // Display the manage page
    public function manageAppliances()
    {
        $appliances = Appliance::all();
        return view('manage', ['appliances' => $appliances]);
    }

    // Handle edit-form submission for an appliance
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

            $appliance = Appliance::findOrFail($id);
            $appliance->update($validatedData);

            return redirect()->route('appliances.manage')
                ->with('success', 'Appliance updated successfully.');
        } catch (\Illuminate\Validation\ValidationException $e) {
            \Log::error('Validation Failed:', $e->errors());
            return redirect()->back()->withErrors($e->errors())->withInput();
        }
    }

    // Update  appliance details
    public function updateAppliance(Request $request, $id)
    {
        \Log::info('updateAppliance() triggered');
        $data = $request->validate([
            'preferred_start' => 'required|date_format:H:i',
            'preferred_end' => 'required|date_format:H:i',
            'duration' => 'required|numeric',
        ]);

        $appliance = Appliance::findOrFail($id);
        $appliance->update($data);

        return response()->json(['success' => true]);
    }

    // Add a selected appliance to schedule day
    public function addSelectedAppliance(Request $request)
    {
        $data = $request->validate([
            'appliance_id' => 'required|integer',
            'preferred_start' => 'required|date_format:H:i',
            'preferred_end' => 'required|date_format:H:i',
            'duration' => 'required|numeric',
            'usage_days' => 'required|in:Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday',
        ]);

        $app = Appliance::find($data['appliance_id']);
        if (!$app) {
            return response()->json(['success' => false, 'message' => 'Appliance not found'], 404);
        }

        // Update appliance preferred times/duration
        $app->update($request->only(['preferred_start', 'preferred_end', 'duration']));

        // Create the SelectedAppliance record
        $sel = SelectedAppliance::create([
            'appliance_id' => $app->id,
            'name' => $app->name,
            'power' => $app->power,
            'preferred_start' => $data['preferred_start'],
            'preferred_end' => $data['preferred_end'],
            'duration' => $data['duration'],
            'usage_days' => $data['usage_days'],
        ]);

        return response()->json(['success' => true, 'selected_appliance_id' => $sel->id]);
    }

    // Get details for one selected appliance
    public function getSelectedAppliance($id)
    {
        $sa = SelectedAppliance::find($id);
        if (!$sa) {
            return response()->json(['success' => false, 'message' => 'Appliance not found'], 404);
        }
        return response()->json(['success' => true, 'appliance' => $sa]);
    }


    // Update a selected appliance  
    public function updateSelectedAppliance(Request $request, $id)
    {
        \Log::info('Update request received:', $request->all());
        $data = $request->validate([
            'preferred_start' => 'required|date_format:H:i',
            'preferred_end' => 'required|date_format:H:i',
            'duration' => 'required|numeric|min:0.01',
        ]);

        $sa = SelectedAppliance::find($id);
        if (!$sa) {
            return response()->json(['success' => false, 'message' => 'Appliance not found'], 404);
        }

        $sa->update($data);
        return response()->json(['success' => true]);
    }

    // Remove a selected appliance
    public function removeSelectedAppliance($id)
    {
        try {
            $sa = SelectedAppliance::findOrFail($id);
            $sa->delete();
            return response()->json(['success' => true]);
        } catch (\Exception $e) {
            \Log::error('Error removing appliance:', ['error' => $e->getMessage()]);
            return response()->json(['success' => false, 'message' => 'An error occurred'], 500);
        }
    }

    // Create the schedule
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
