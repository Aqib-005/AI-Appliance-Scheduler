<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use App\Models\Appliance;
use App\Models\SelectedAppliance;
use App\Models\Schedule;

class ScheduleController extends Controller
{
    public function dashboard()
    {
        $predictions = session('predictions', []);
        $pricesByDayHour = [];
        foreach ($predictions as $prediction) {
            $dt = \Carbon\Carbon::parse($prediction['StartDateTime']);
            $day = $dt->format('l');
            $hour = (int) $dt->format('H');
            $pricesByDayHour[$day][$hour] = $prediction['Predicted_Price'];
        }

        $appliances = Appliance::all();
        $selectedAppliances = SelectedAppliance::with('appliance')->get();
        $schedule = Schedule::with('appliance')->get();

        $scheduledLoadByDayHour = [];
        $weeklyCost = 0;

        foreach ($schedule as $s) {
            $duration = $s->end_hour - $s->start_hour;
            $power = $s->appliance ? $s->appliance->power : 0;
            for ($h = $s->start_hour; $h < $s->end_hour; $h++) {
                if (!isset($scheduledLoadByDayHour[$s->day])) {
                    $scheduledLoadByDayHour[$s->day] = [];
                }
                $scheduledLoadByDayHour[$s->day][$h] = ($scheduledLoadByDayHour[$s->day][$h] ?? 0) + $power;
            }

            $costForSchedule = 0;
            for ($h = $s->start_hour; $h < $s->end_hour; $h++) {
                if (isset($pricesByDayHour[$s->day][$h])) {
                    $pricePerKWh = $pricesByDayHour[$s->day][$h] / 1000;
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
                $baselineLoadByDayHour[$s['day']][$h] = ($baselineLoadByDayHour[$s['day']][$h] ?? 0) + $power;
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

        \Log::info("RESULTS:");
        \Log::info("Baseline PAR: " . number_format($baselinePAR, 2));
        \Log::info("Scheduled PAR: " . number_format($scheduledPAR, 2));

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
                    $allLoads[] = $load;
            }
        }

        if (empty($allLoads))
            return 0;

        $peak = max($allLoads);
        $average = array_sum($allLoads) / count($allLoads);
        return $peak / $average;
    }

    private function getBaselineSchedule($appliances)
    {
        $applianceStartTimes = [
            'Oven' => 6,
            'Dishwasher' => 12,
            'Washing Machine' => 14,
            'Electric Heater' => 15,
            'Vacuum' => 15
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
                'name' => $name
            ];
        }

        return $baselineSchedule;
    }

    public function showResults()
    {
        $schedule = session('schedule');
        $predictions = session('predictions');

        return view('results', [
            'schedule' => $schedule,
            'predictions' => $predictions,
        ]);
    }

    public function store(Request $request)
    {
        set_time_limit(300);
        Schedule::truncate();
        $start_date = date('Y-m-d', strtotime('monday this week'));

        \Log::info('Calculated start_date:', ['start_date' => $start_date]);

        try {
            $start_time_ml = microtime(true);  // Start timing the ML prediction API call
            $response = Http::timeout(300)->post('http://127.0.0.1:8001/predict', [
                'start_date' => $start_date,
            ]);
            $end_time_ml = microtime(true);  // End timing the ML prediction API call
            $runtime_ml = $end_time_ml - $start_time_ml;
            \Log::info("ML prediction runtime: " . number_format($runtime_ml, 2) . " seconds");

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

            $predictions = $response->json()['predictions'];
            session(['predictions' => $predictions]);

            $appliances = SelectedAppliance::all()->map(function ($selectedAppliance) {
                return [
                    'id' => $selectedAppliance->appliance_id,
                    'name' => $selectedAppliance->name,
                    'power' => $selectedAppliance->power,
                    'preferred_start' => $selectedAppliance->preferred_start,
                    'preferred_end' => $selectedAppliance->preferred_end,
                    'duration' => $selectedAppliance->duration,
                    'usage_days' => $selectedAppliance->usage_days,
                ];
            })->toArray();

            $start_time_scheduling = microtime(true);  // Start timing the scheduling algorithm
            $this->scheduleAppliances($predictions, $appliances);
            $end_time_scheduling = microtime(true);  // End timing the scheduling algorithm
            $runtime_scheduling = $end_time_scheduling - $start_time_scheduling;
            \Log::info("Scheduling algorithm runtime: " . number_format($runtime_scheduling, 2) . " seconds");

            return response()->json([
                'success' => true,
                'redirect_url' => route('dashboard'),
            ]);
        } catch (\Exception $e) {
            \Log::error('Error in store method:', ['error' => $e->getMessage()]);
            return response()->json(['success' => false, 'message' => 'An error occurred. Please try again.'], 500);
        }
    }

    private function scheduleAppliances($predictions, $appliances)
    {
        set_time_limit(300);
        \Log::info('Predictions:', $predictions);
        \Log::info('Appliances:', $appliances);

        usort($predictions, function ($a, $b) {
            return (float) $a['Predicted_Price'] <=> (float) $b['Predicted_Price'];
        });

        $schedule = [];
        $predictionsByDay = [];
        foreach ($predictions as $prediction) {
            $day = date('l', strtotime($prediction['StartDateTime']));
            $hour = (int) date('H', strtotime($prediction['StartDateTime']));
            $predictionsByDay[$day][$hour] = $prediction;
        }

        foreach ($predictionsByDay as $day => &$hours) {
            ksort($hours);
            $prices = array_column($hours, 'Predicted_Price');
            $medianPrice = $this->calculatePercentile($prices, 50);
            $peakThreshold = $this->calculatePercentile($prices, 75);

            foreach ($hours as $hour => $data) {
                $price = (float) $data['Predicted_Price'];
                $hours[$hour]['isPeak'] = ($price >= $peakThreshold);
                $hours[$hour]['peakPenalty'] = max(0, ($price - $medianPrice) / $medianPrice);
            }
        }

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

            $windows = $this->findWindows(
                $predictionsByDay[$day],
                $schedule[$day] ?? [],
                $preferredStart,
                $preferredEnd,
                $duration
            );

            if (empty($windows)) {
                \Log::info('No preferred windows found, searching full day', [
                    'appliance_id' => $appliance['id']
                ]);

                $windows = $this->findWindows(
                    $predictionsByDay[$day],
                    $schedule[$day] ?? [],
                    0,
                    23 - $duration + 1,
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

    public function removeAppliance($id)
    {
        $appliance = Appliance::find($id);

        if ($appliance) {
            $appliance->delete();
            return response()->json(['success' => true]);
        }

        return response()->json(['success' => false], 400);
    }

    public function manageAppliances()
    {
        $appliances = Appliance::all();
        return view('manage', ['appliances' => $appliances]);
    }

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

            $appliance->update([
                'name' => $validatedData['name'],
                'power' => $validatedData['power'],
                'preferred_start' => $validatedData['preferred_start'],
                'preferred_end' => $validatedData['preferred_end'],
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

    public function updateAppliance(Request $request, $id)
    {
        \Log::info('updateAppliance() triggered');
        $request->validate([
            'preferred_start' => 'required|date_format:H:i',
            'preferred_end' => 'required|date_format:H:i',
            'duration' => 'required|numeric',
        ]);

        $appliance = Appliance::findOrFail($id);
        $appliance->update([
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
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

        $appliance = Appliance::find($request->input('appliance_id'));
        if (!$appliance) {
            return response()->json(['success' => false, 'message' => 'Appliance not found.'], 404);
        }

        $appliance->update([
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
        ]);

        $selectedAppliance = SelectedAppliance::create([
            'appliance_id' => $request->input('appliance_id'),
            'name' => $appliance->name,
            'power' => $appliance->power,
            'preferred_start' => $request->input('preferred_start'),
            'preferred_end' => $request->input('preferred_end'),
            'duration' => $request->input('duration'),
            'usage_days' => $request->input('usage_days'),
        ]);

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
            $selectedAppliance = SelectedAppliance::find($id);

            if (!$selectedAppliance) {
                return response()->json(['success' => false, 'message' => 'Appliance not found.'], 404);
            }

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