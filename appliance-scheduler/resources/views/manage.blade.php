<!DOCTYPE html>
<html>

<head>
    <title>Manage Appliances</title>
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
    </style>
</head>

<body>
    <h1>Manage Appliances</h1>
    <a href="{{ route('dashboard') }}">
        <button>Back to Dashboard</button>
    </a>

    <!-- Add Appliance Form -->
    <h2>Add Appliance</h2>
    <form action="{{ route('appliance.add') }}" method="POST">
        @csrf
        <label for="name">Appliance Name:</label>
        <input type="text" id="name" name="name" required>
        <br>
        <label for="power">Power (kW):</label>
        <input type="number" step="0.1" id="power" name="power" required>
        <br>
        <label for="preferred_start">Preferred Start Hour (0-23):</label>
        <input type="number" id="preferred_start" name="preferred_start" min="0" max="23" required>
        <br>
        <label for="preferred_end">Preferred End Hour (0-23):</label>
        <input type="number" id="preferred_end" name="preferred_end" min="0" max="23" required>
        <br>
        <label for="duration">Duration (hours):</label>
        <input type="number" step="0.1" id="duration" name="duration" required>
        <br>
        <button type="submit">Add Appliance</button>
    </form>

    <!-- List of Appliances with Edit/Delete Options -->
    <h2>Existing Appliances</h2>
    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Power (kW)</th>
                <th>Preferred Start</th>
                <th>Preferred End</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            @foreach ($appliances as $appliance)
                <tr>
                    <td>{{ $appliance->name }}</td>
                    <td>{{ $appliance->power }}</td>
                    <td>{{ $appliance->preferred_start }}</td>
                    <td>{{ $appliance->preferred_end }}</td>
                    <td>
                        <a href="{{ route('appliance.edit', $appliance->id) }}">
                            <button>Edit</button>
                        </a>
                        <form action="{{ route('appliance.remove', $appliance->id) }}" method="POST"
                            style="display:inline;">
                            @csrf
                            @method('DELETE')
                            <button type="submit">Delete</button>
                        </form>
                    </td>
                </tr>
            @endforeach
        </tbody>
    </table>
</body>

</html>