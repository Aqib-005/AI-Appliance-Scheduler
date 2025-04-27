<!DOCTYPE html>
<html>

<head>
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <title>Manage Appliances</title>
    <style>
        :root {
            --border: 1px solid #ccc;
            --bg-header: #f5f5f5;
            --bg-cell: #fff;
            --text: #000;
            --btn-padding: 10px 20px;
            --btn-radius: 5px;
            --btn-font: 1em;
            --color-primary: #007bff;
            --color-primary-hover: #0056b3;
            --color-success: #218838;
            --color-success-hover: #1e7e34;
        }

        body {
            margin: 0;
            padding: 20px;
            background: var(--bg-cell);
            color: var(--text);
            font-family: Arial, sans-serif;
        }

        .header,
        .button-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .header h1,
        .button-container h2 {
            margin: 0;
        }

        .btn {
            padding: var(--btn-padding);
            border: none;
            border-radius: var(--btn-radius);
            cursor: pointer;
            font-size: var(--btn-font);
        }

        .btn-primary {
            background-color: var(--color-primary);
            color: #fff;
        }

        .btn-primary:hover {
            background-color: var(--color-primary-hover);
        }

        .btn-success {
            background-color: var(--color-success);
            color: #fff;
        }

        .btn-success:hover {
            background-color: var(--color-success-hover);
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th,
        td {
            border: var(--border);
            padding: 8px;
            text-align: left;
        }

        th {
            background-color: var(--bg-header);
        }

        .popup {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--bg-cell);
            padding: 20px;
            border-radius: var(--btn-radius);
            box-shadow: 0 0 10px rgba(0, 0, 0, .1);
            z-index: 1000;
            width: 320px;
            max-width: 90%;
        }

        .overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, .5);
            z-index: 999;
        }

        .popup.active,
        .overlay.active {
            display: block;
        }

        .popup label {
            display: block;
            margin-top: 10px;
        }

        .popup input {
            width: 100%;
            padding: 6px;
            margin-top: 4px;
            box-sizing: border-box;
        }

        .popup .actions {
            margin-top: 20px;
            text-align: right;
        }

        .popup .actions .btn {
            margin-left: 10px;
        }
    </style>
</head>

<body>
    <div class="header">
        <h1>Manage Appliances</h1>
        <a href="{{ route('dashboard') }}">
            <button class="btn btn-success">Back to Dashboard</button>
        </a>
    </div>

    <div class="button-container">
        <h2>Existing Appliances</h2>
        <button class="btn btn-success" onclick="openAddPopup()">Add Appliance</button>
    </div>

    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Power (kW/h)</th>
                <th>Preferred Start</th>
                <th>Preferred End</th>
                <th>Duration (hours)</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            @foreach ($appliances as $appliance)
                <tr data-id="{{ $appliance->id }}">
                    <td>{{ $appliance->name }}</td>
                    <td>{{ $appliance->power }}</td>
                    <td>{{ date('H:i', strtotime($appliance->preferred_start)) }}</td>
                    <td>{{ date('H:i', strtotime($appliance->preferred_end)) }}</td>
                    <td>{{ $appliance->duration }}</td>
                    <td>
                        <button class="btn btn-primary" onclick="openEditPopup({{ $appliance->id }})">
                            Edit
                        </button>
                        <button class="btn btn-primary" onclick="openDeletePopup({{ $appliance->id }})">
                            Delete
                        </button>
                    </td>
                </tr>
            @endforeach
        </tbody>
    </table>

    <!-- Add Appliance Popup -->
    <div id="addPopup" class="popup">
        <h2>Add Appliance</h2>
        <form action="{{ route('appliance.add') }}" method="POST">
            @csrf
            <label for="name">Appliance Name:</label>
            <input type="text" id="name" name="name" required>
            <label for="power">Power (kW/h):</label>
            <input type="number" step="0.01" id="power" name="power" required>
            <label for="preferred_start">Preferred Start Time:</label>
            <input type="time" id="preferred_start" name="preferred_start" required>
            <label for="preferred_end">Preferred End Time:</label>
            <input type="time" id="preferred_end" name="preferred_end" required>
            <label for="duration">Duration (hours):</label>
            <input type="number" step="0.1" id="duration" name="duration" required>
            <div class="actions">
                <button type="submit" class="btn btn-primary">Add</button>
                <button type="button" class="btn" onclick="closeAddPopup()">Cancel</button>
            </div>
        </form>
    </div>

    <!-- Edit Appliance Popup -->
    <div id="editPopup" class="popup">
        <h2>Edit Appliance</h2>
        <form id="editForm" method="POST">
            @csrf
            @method('PUT')
            <label for="edit_name">Appliance Name:</label>
            <input type="text" id="edit_name" name="name" required>
            <label for="edit_power">Power (kW/h):</label>
            <input type="number" step="0.01" id="edit_power" name="power" required>
            <label for="edit_preferred_start">Preferred Start Time:</label>
            <input type="time" id="edit_preferred_start" name="preferred_start" required>
            <label for="edit_preferred_end">Preferred End Time:</label>
            <input type="time" id="edit_preferred_end" name="preferred_end" required>
            <label for="edit_duration">Duration (hours):</label>
            <input type="number" step="0.1" id="edit_duration" name="duration" required>
            <div class="actions">
                <button type="submit" class="btn btn-primary">Update</button>
                <button type="button" class="btn" onclick="closeEditPopup()">Cancel</button>
            </div>
        </form>
    </div>

    <!-- Delete Confirmation Popup -->
    <div id="deletePopup" class="popup">
        <h2>Delete Appliance</h2>
        <p>Are you sure you want to delete this appliance?</p>
        <div class="actions">
            <button id="confirmDeleteBtn" class="btn btn-success">Yes</button>
            <button class="btn" onclick="closeDeletePopup()">No</button>
        </div>
    </div>

    <div id="overlay" class="overlay"></div>

    <script>
        let pendingDeleteId = null;

        // == Add popup
        function openAddPopup() {
            document.getElementById('addPopup').classList.add('active');
            document.getElementById('overlay').classList.add('active');
        }
        function closeAddPopup() {
            document.getElementById('addPopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        // == Edit popup
        function openEditPopup(id) {
            const app = {!! json_encode($appliances->keyBy('id')->toArray()) !!}[id];
            document.getElementById('edit_name').value = app.name;
            document.getElementById('edit_power').value = app.power;
            document.getElementById('edit_preferred_start').value = app.preferred_start?.substr(0, 5);
            document.getElementById('edit_preferred_end').value = app.preferred_end?.substr(0, 5);
            document.getElementById('edit_duration').value = app.duration;
            document.getElementById('editForm').action = `/appliance/edit/${id}`;

            document.getElementById('editPopup').classList.add('active');
            document.getElementById('overlay').classList.add('active');
        }
        function closeEditPopup() {
            document.getElementById('editPopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        // == Delete popup
        function openDeletePopup(id) {
            pendingDeleteId = id;
            document.getElementById('deletePopup').classList.add('active');
            document.getElementById('overlay').classList.add('active');
        }
        function closeDeletePopup() {
            document.getElementById('deletePopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        // == AJAX Delete
        document.getElementById('confirmDeleteBtn').addEventListener('click', () => {
            if (!pendingDeleteId) return closeDeletePopup();

            fetch(`/appliance/remove/${pendingDeleteId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content
                }
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        // remove the row
                        const row = document.querySelector(`tr[data-id="${pendingDeleteId}"]`);
                        if (row) row.remove();
                    } else {
                        alert('Delete failed: ' + (data.message || 'unknown error'));
                    }
                })
                .catch(e => {
                    console.error(e);
                    alert('Error deleting appliance.');
                })
                .finally(() => {
                    pendingDeleteId = null;
                    closeDeletePopup();
                });
        });
    </script>
</body>

</html>