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

        /* Added container for the timetable table */
        .table-container {
            flex: 1;
            overflow-y: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th,
        td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        th {
            background-color: #f2f2f2;
        }

        .button-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        button {
            padding: 10px 20px;
            cursor: pointer;
        }

        .current-hour {
            border-left: 3px solid #007bff;
        }
    </style>
</head>

<body>
    <h1>Dashboard</h1>

    <div class="container">
        <!-- Left Side: Timetable -->
        <div class="left">
            <div class="button-container">
                <h2>Scheduled Appliances</h2>
                <a href="{{ route('schedule.create') }}">
                    <button>Schedule</button>
                </a>
            </div>
            <!-- Wrap the timetable in a container that scrolls if needed -->
            <div class="table-container">
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
                            <tr>
                                <td>{{ sprintf('%02d:00', $hour) }}</td>
                                @foreach (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
                                    <td>
                                        @foreach ($schedule as $entry)
                                            @if ($entry->day === $day && $hour >= $entry->start_hour && $hour < $entry->end_hour)
                                                {{ $entry->appliance->name ?? 'Appliance Not Found' }}
                                            @endif
                                        @endforeach
                                    </td>
                                @endforeach
                            </tr>
                        @endfor
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Right Side: Appliances and Prices -->
        <div class="right">
            <!-- Appliances List -->
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

            <!-- Predicted Prices (Line Chart) -->
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

                                <!-- Include Chart.js and the Moment adapter -->
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
        document.addEventListener('DOMContentLoaded', function () {
            const now = new Date();
            const currentHour = now.getHours();
            const table = document.querySelector('.table-container table');
            const rows = table.querySelectorAll('tbody tr');
            if (rows[currentHour]) {
                rows[currentHour].classList.add('current-hour');
                rows[currentHour].scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });

        document.addEventListener('DOMContentLoaded', function () {
            const now = new Date();
            const currentHour = now.getHours();
            const table = document.querySelector('.table-container table');
            const rows = table.querySelectorAll('tbody tr');
            if (rows[currentHour]) {
                rows[currentHour].classList.add('current-hour');
                rows[currentHour].scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });
    </script>
</body>

</html>