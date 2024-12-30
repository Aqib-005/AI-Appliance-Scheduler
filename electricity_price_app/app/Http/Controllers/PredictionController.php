<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;

class PredictionController extends Controller
{
    // Display scheduling form
    public function index()
    {
        return view('schedule');
    }

    // Handle  scheduling
    public function schedule(Request $request)
    {
        // Validate  user input
        $request->validate([
            'appliances' => 'required|array',
            'appliances.*.name' => 'required|string',
            'appliances.*.consumption' => 'required|numeric',
            'appliances.*.preferred_time' => 'required|string',
        ]);

        // Get the appliances data
        $appliances = $request->appliances;

        //  get predicted prices
        $process = new Process(['C:\\Python312\\python.exe', base_path('ml_models/predict.py')]);
        $process->run();

        if (!$process->isSuccessful()) {
            throw new ProcessFailedException($process);
        }

        $predictedPrices = json_decode($process->getOutput(), true);

        // Call the scheduling algorithm
        $schedule = $this->scheduleAppliances($appliances, $predictedPrices);

        // Show the schedule to the user
        return view('schedule', ['schedule' => $schedule]);
    }

    // Greedy scheduling algorithm
    private function scheduleAppliances($appliances, $predictedPrices)
    {
        // Sort appliances by consumption (highest first)
        usort($appliances, function ($a, $b) {
            return $b['consumption'] <=> $a['consumption'];
        });

        // Sort predicted prices by cost (lowest first)
        asort($predictedPrices);

        // Assign appliances to the cheapest times within their preferred times
        $schedule = [];
        foreach ($appliances as $appliance) {
            foreach ($predictedPrices as $time => $price) {
                if ($this->isPreferredTime($time, $appliance['preferred_time'])) {
                    $schedule[] = [
                        'appliance' => $appliance['name'],
                        'time' => $time,
                        'price' => $price,
                    ];
                    unset($predictedPrices[$time]); // Remove this time slot
                    break;
                }
            }
        }

        return $schedule;
    }

    // Check if a time slot matches the preferred time
    private function isPreferredTime($time, $preferredTime)
    {
        // Split the preferred time into start and end
        list($start, $end) = explode('-', $preferredTime);

        // Convert times to minutes for easy comparison
        $startMinutes = $this->timeToMinutes($start);
        $endMinutes = $this->timeToMinutes($end);
        $timeMinutes = $this->timeToMinutes($time);

        // Check if the time is within the preferred range
        return $timeMinutes >= $startMinutes && $timeMinutes <= $endMinutes;
    }

    // Convert to minutes
    private function timeToMinutes($time)
    {
        list($hours, $minutes) = explode(':', $time);
        return (int) $hours * 60 + (int) $minutes;
    }
}