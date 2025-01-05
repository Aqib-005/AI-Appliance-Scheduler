<!DOCTYPE html>
<html>

<head>
    <title>Schedule Appliances</title>
    <style>
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
    <h1>Schedule Appliances</h1>
    <a href="{{ route('dashboard') }}">
        <button>Back to Dashboard</button>
    </a>

    <form action="{{ route('schedule.store') }}" method="POST">
        @csrf
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
                @foreach (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
                    <tr>
                        <td>{{ $day }}</td>
                        <td>
                            <select name="appliance_{{ strtolower($day) }}" required>
                                <option value="">Select Appliance</option>
                                @foreach ($appliances as $appliance)
                                    <option value="{{ $appliance->id }}">{{ $appliance->name }}</option>
                                @endforeach
                            </select>
                        </td>
                        <td>
                            <input type="number" name="start_hour_{{ strtolower($day) }}" min="0" max="23" required>
                        </td>
                        <td>
                            <input type="number" name="end_hour_{{ strtolower($day) }}" min="0" max="23" required>
                        </td>
                    </tr>
                @endforeach
            </tbody>
        </table>
        <button type="submit">Save Schedule</button>
    </form>
</body>

</html>