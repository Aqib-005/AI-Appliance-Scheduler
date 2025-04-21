<!DOCTYPE html>
<html>

<head>
    <title>Dashboard</title>
    <style>
        .container {
            display: flex;
        }

        .left {
            flex: 60%;
            padding: 10px;
            display: flex;
            flex-direction: column;
        }

        .right {
            flex: 40%;
            padding: 10px;
        }

        .window {
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 20px;
        }

        .table-container {
            flex: 1;
            height: 400px;
            overflow-y: auto;
            overflow-x: auto;
            border: 1px solid #ddd;
            position: relative;
            width: 100%;
            box-sizing: border-box;
            background-color: #2e2e2e;
        }

        table {
            width: 100%;
            min-width: 800px;
            border-collapse: collapse;
        }

        th,
        td {
            border: 1px solid #444;
            padding: 0;
            text-align: center;
            vertical-align: top;
            height: 30px;
            color: #fff;
        }

        th {
            background-color: #1e1e1e;
            position: sticky;
            top: 0;
            z-index: 1;
            color: #fff;
        }

        td {
            background-color: #2e2e2e;
        }

        .appliance-cell {
            display: flex;
            flex-direction: row;
            height: 100%;
            width: 100%;
        }

        .appliance-block {
            flex: 1;
            background-color: #26a69a;
            border: 1px solid #4dd0e1;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-size: 0.9em;
            padding: 5px;
            box-sizing: border-box;
            min-height: 100%;
        }

        .button-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }

        button {
            padding: 10px 20px;
            cursor: pointer;
        }

        .current-hour {
            border-left: 3px solid #007bff;
            background-color: #3e3e3e;
        }

        .weekly-cost {
            margin-bottom: 10px;
            font-size: 1.2em;
            font-weight: bold;
        }

        .table-container.fullscreen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 9999;
            background-color: #2e2e2e;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
            overflow-y: auto;
            overflow-x: auto;
        }

        #exitFullscreenBtn {
            display: none;
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 10001;
            font-size: 1.2em;
            background-color: #dc3545;
            color: #fff;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            cursor: pointer;
        }

        td:empty {
            background-color: #2e2e2e;
        }
    </style>
</head>

<body>
    <h1>Dashboard</h1>

    <div class="container">
        <!-- Left Side: Timetable and Weekly Cost -->
        <div class="left">
            <div class="button-container">
                <h2 style="margin: 0;">Scheduled Appliances</h2>
                <a href="{{ route('schedule.create') }}">
                    <button>Schedule</button>
                </a>
                <button id="toggleFullscreenBtn">Fullscreen</button>
            </div>

            <div class="weekly-cost">
                Weekly Cost: €{{ number_format($weeklyCost, 2) }}
            </div>

            <div class="table-container" id="timetableContainer">
                <button id="exitFullscreenBtn">X</button>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            @foreach (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
                                <th>{{ $day }}</th>
                            @endforeach
                        </tr>
                    </thead>
                    <tbody>
                        @for ($hour = 0; $hour < 24; $hour++)
                                            <tr id="row-{{ $hour }}" data-hour="{{ $hour }}">
                                                <td>{{ sprintf('%02d:00', $hour) }}</td>
                                                @foreach (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
                                                                        @php
                                                                            $appliancesInSlot = [];
                                                                            $isWithinDuration = false;

                                                                            // Check if this hour is within the duration of an earlier appliance
                                                                            foreach ($schedule as $entry) {
                                                                                if ($entry->day === $day && $hour > $entry->start_hour && $hour < $entry->end_hour) {
                                                                                    $isWithinDuration = true;
                                                                                    break;
                                                                                }
                                                                            }

                                                                            // Collect appliances starting at this hour
                                                                            if (!$isWithinDuration) {
                                                                                foreach ($schedule as $entry) {
                                                                                    if ($entry->day === $day && $entry->start_hour == $hour) {
                                                                                        $duration = $entry->end_hour - $entry->start_hour;
                                                                                        $appliancesInSlot[] = [
                                                                                            'name' => $entry->appliance->name ?? 'Appliance Not Found',
                                                                                            'duration' => $duration
                                                                                        ];
                                                                                    }
                                                                                }

                                                                                // Render the cell
                                                                                if (!empty($appliancesInSlot)) {
                                                                                    // Use the duration of the first appliance for rowspan (we'll handle side-by-side rendering)
                                                                                    $firstApplianceDuration = $appliancesInSlot[0]['duration'];
                                                                                    echo '<td rowspan="' . $firstApplianceDuration . '" class="appliance-cell">';
                                                                                    foreach ($appliancesInSlot as $appliance) {
                                                                                        // Adjust height to match the appliance's duration
                                                                                        $blockHeight = $appliance['duration'] * 30; // 30px per hour
                                                                                        echo '<div class="appliance-block" style="height: ' . $blockHeight . 'px;">' . $appliance['name'] . '</div>';
                                                                                    }
                                                                                    echo '</td>';
                                                                                } else {
                                                                                    echo '<td></td>';
                                                                                }
                                                                            }
                                                                        @endphp
                                                @endforeach
                                            </tr>
                        @endfor
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Right Side: Appliances and Prices -->
        <div class="right">
            <div class="window">
                <h2>Appliances</h2>
                <ul>
                    @foreach ($appliances as $appliance)
                        <li>{{ $appliance->name }} ({{ $appliance->power }} kW)</li>
                    @endforeach
                </ul>
                <a href="{{ route('appliances.manage') }}">
                    <button>View All Appliances</button>
                </a>
            </div>

            <div class="window">
                <h2>Predicted Prices</h2>
                @if (!empty($predictions))
                                @php
                                    $chartData = [];
                                    foreach ($predictions as $prediction) {
                                        $datetime = \Carbon\Carbon::parse($prediction['StartDateTime'])->toDateTimeString();
                                        $chartData[] = [
                                            'x' => $datetime,
                                            'y' => $prediction['Predicted_Price']
                                        ];
                                    }
                                @endphp

                                <canvas id="predictionsChart"></canvas>

                                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                                <script src="https://cdn.jsdelivr.net/npm/moment@2.29.3/moment.min.js"></script>
                                <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-moment"></script>

                                <script>
                                    const ctx = document.getElementById('predictionsChart').getContext('2d');
                                    const chartData = {!! json_encode($chartData) !!};

                                    new Chart(ctx, {
                                        type: 'line',
                                        data: {
                                            datasets: [{
                                                label: 'Predicted Price (€/MWh)',
                                                data: chartData,
                                                borderColor: 'rgba(75, 192, 192, 1)',
                                                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                                fill: false,
                                                tension: 0.1
                                            }]
                                        },
                                        options: {
                                            scales: {
                                                x: {
                                                    type: 'time',
                                                    time: {
                                                        unit: 'day',
                                                        displayFormats: {
                                                            day: 'YYYY-MM-DD'
                                                        }
                                                    },
                                                },
                                                y: {
                                                    beginAtZero: false
                                                }
                                            }
                                        }
                                    });
                                </script>
                @else
                    <p>No predictions available.</p>
                @endif
            </div>
        </div>
    </div>

    <script>
        function scrollToCurrentHour() {
            const tableContainer = document.getElementById('timetableContainer');
            const currentRow = document.getElementById(`row-${new Date().getHours()}`);
            if (currentRow) {
                currentRow.classList.add('current-hour');
                tableContainer.scrollTop = currentRow.offsetTop - tableContainer.clientHeight / 2;
            }
        }

        document.addEventListener('DOMContentLoaded', function () {
            scrollToCurrentHour();

            const toggleFullscreenBtn = document.getElementById('toggleFullscreenBtn');
            const exitFullscreenBtn = document.getElementById('exitFullscreenBtn');
            const tableContainer = document.getElementById('timetableContainer');

            toggleFullscreenBtn.addEventListener('click', () => {
                tableContainer.classList.add('fullscreen');
                exitFullscreenBtn.style.display = 'block';
            });

            exitFullscreenBtn.addEventListener('click', () => {
                tableContainer.classList.remove('fullscreen');
                exitFullscreenBtn.style.display = 'none';
                scrollToCurrentHour();
            });
        });
    </script>
</body>

</html>