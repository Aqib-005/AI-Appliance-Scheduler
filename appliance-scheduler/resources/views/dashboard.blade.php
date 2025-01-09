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
            <table>
                <thead>
                    <tr>
                        <th>Day</th>
                        <th>Appliance</th>
                        <th>Start Hour</th>
                        <th>End Hour</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach ($schedules as $schedule)
                        <tr>
                            <td>{{ $schedule->day }}</td>
                            <td>{{ $schedule->appliance->name }}</td>
                            <td>{{ $schedule->start_hour }}:00</td>
                            <td>{{ $schedule->end_hour }}:00</td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
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

            <!-- Predicted Prices -->
            <div class="window">
                <h2>Predicted Prices</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Start Date/Time</th>
                            <th>Predicted Price (€/MWh)</th>
                        </tr>
                    </thead>
                    <tbody>
                        @if (isset($predictions))
                            @foreach ($predictions as $prediction)
                                <tr>
                                    <td>{{ $prediction['Start date/time'] }}</td>
                                    <td>{{ $prediction['Predicted Price [Euro/MWh]'] }}</td>
                                </tr>
                            @endforeach
                        @else
                            <tr>
                                <td colspan="2">No predictions available.</td>
                            </tr>
                        @endif
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>

</html>