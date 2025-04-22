<!DOCTYPE html>
<html>

<head>
    <title>Dashboard</title>
    <style>
        :root {
            --first-col: 60px;
            /* time label width */
            --row-height: 30px;
            /* height per hour */
            --days: 7;
            /* Monday→Sunday */
            --border: 1px solid #ccc;

            --bg-cell: #fff;
            --bg-header: #f5f5f5;
            --text: #000;

            /* col‑width for each day */
            --col-width: calc((100% - var(--first-col)) / var(--days));
        }

        body {
            margin: 0;
            padding: 20px;
            background: var(--bg-cell);
            color: var(--text);
            font-family: sans-serif;
        }

        h1,
        h2 {
            margin: 0;
        }

        .container {
            display: flex;
            gap: 20px;
        }

        .left {
            flex: 60%;
            display: flex;
            flex-direction: column;
        }

        .button-container {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }

        .weekly-cost {
            margin-bottom: 10px;
            font-size: 1.2em;
            font-weight: bold;
        }

        .right {
            flex: 40%;
        }

        /* wrapper for grid + blocks */
        .timetable-wrapper {
            position: relative;
            border: var(--border);
            overflow: auto;
            height: calc(var(--row-height) * 25);
            background: var(--bg-cell);
        }

        /* the 8×25 grid */
        .timetable-grid {
            display: grid;
            grid-template-columns: var(--first-col) repeat(var(--days), 1fr);
            grid-template-rows: var(--row-height) repeat(24, var(--row-height));
        }

        .timetable-grid>div {
            box-sizing: border-box;
            border: var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-cell);
            z-index: 1;
        }

        .timetable-grid .header {
            background: var(--bg-header);
            position: sticky;
            top: 0;
            z-index: 3;
        }

        .timetable-grid .time-label {
            background: var(--bg-header);
            position: sticky;
            left: 0;
            z-index: 3;
        }

        /* overlay for blocks */
        .blocks-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .appliance-block {
            position: absolute;
            box-sizing: border-box;
            padding: 4px;
            font-size: 0.85em;
            color: var(--text);
            border-radius: 3px;
            text-align: center;
            /* allow wrapping so full names show */
            white-space: normal;
            word-break: break-word;
            /* let text expand vertically if needed */
            overflow: visible;
            pointer-events: auto;
            z-index: 2;
        }
    </style>
</head>

<body>
    <h1>Dashboard</h1>
    <div class="container">
        <div class="left">
            <div class="button-container">
                <h2>Scheduled Appliances</h2>
                <a href="{{ route('schedule.create') }}"><button>Schedule</button></a>
                <button id="toggleFullscreenBtn">Fullscreen</button>
            </div>
            <div class="weekly-cost">
                Weekly Cost: €{{ number_format($weeklyCost, 2) }}
            </div>

            <div class="timetable-wrapper" id="timetableContainer">
                {{-- GRID OF CELLS --}}
                <div class="timetable-grid">
                    <div class="header">Time</div>
                    @php
                        $dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
                      @endphp
                    @foreach($dayNames as $day)
                        <div class="header">{{ $day }}</div>
                    @endforeach

                    @for($h = 0; $h < 24; $h++)
                        <div class="time-label">{{ sprintf('%02d:00', $h) }}</div>
                        @for($d = 0; $d < 7; $d++)
                            <div></div>
                        @endfor
                    @endfor
                </div>

                {{-- GROUP & SORT FOR OVERLAP --}}
                @php
                    $byDay = [];
                    foreach ($schedule as $e) {
                        $byDay[$e->day][] = $e;
                    }
                    foreach ($byDay as &$list) {
                        usort($list, fn($a, $b) => $a->start_hour <=> $b->start_hour);
                    }
                @endphp

                {{-- BLOCKS OVERLAY --}}
                <div class="blocks-container">
                    @foreach($byDay as $day => $entries)
                                    @php $d = array_search($day, $dayNames); @endphp
                                    @foreach($entries as $entry)
                                                    @php
                                                        $s = $entry->start_hour;
                                                        $e = $entry->end_hour;
                                                        $overlaps = array_filter(
                                                            $entries,
                                                            fn($x) =>
                                                            $x->start_hour < $e && $x->end_hour > $s
                                                        );
                                                        usort($overlaps, fn($a, $b) => $a->start_hour <=> $b->start_hour);
                                                        $count = count($overlaps);
                                                        $pos = array_search($entry, $overlaps, true);

                                                        // random transparent color:
                                                        $hue = crc32($entry->appliance_id . $s . $day) % 360;
                                                        $bg = "hsla($hue,70%,50%,0.4)";
                                                        $border = "hsla($hue,70%,40%,1)";

                                                        $top = "calc(var(--row-height) * (1 + $s))";
                                                        $height = "calc(var(--row-height) * (" . ($e - $s) . "))";
                                                        $width = "calc(var(--col-width) / $count)";
                                                        $left = "calc(var(--first-col) + var(--col-width) * $d + ($width) * $pos)";
                                                      @endphp

                                                    <div class="appliance-block" style="
                                                     top: {{ $top }};
                                                     height: {{ $height }};
                                                     left: {{ $left }};
                                                     width: {{ $width }};
                                                     background-color: {{ $bg }};
                                                     border: 1px solid {{ $border }};
                                                   ">
                                                        {{ $entry->appliance->name }}
                                                    </div>
                                    @endforeach
                    @endforeach
                </div>
            </div>
        </div>

        <div class="right">
            <div class="window">
                <h2>Appliances</h2>
                <ul>
                    @foreach($appliances as $a)
                        <li>{{ $a->name }} ({{ $a->power }} kW)</li>
                    @endforeach
                </ul>
                <a href="{{ route('appliances.manage') }}"><button>View All Appliances</button></a>
            </div>
            <div class="window">
                <h2>Predicted Prices</h2>
                @if(!empty($predictions))
                                @php
                                    $chartData = collect($predictions)->map(fn($p) => [
                                        'x' => \Carbon\Carbon::parse($p['StartDateTime'])->toDateTimeString(),
                                        'y' => $p['Predicted_Price']
                                    ]);
                                  @endphp
                                <canvas id="predictionsChart"></canvas>
                                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                                <script src="https://cdn.jsdelivr.net/npm/moment@2.29.3/moment.min.js"></script>
                                <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-moment"></script>
                                <script>
                                    const ctx = document.getElementById('predictionsChart').getContext('2d');
                                    new Chart(ctx, {
                                        type: 'line',
                                        data: {
                                            datasets: [{
                                                label: 'Price (€/MWh)',
                                                data: {!! $chartData->toJson() !!},
                                                fill: false,
                                                tension: 0.1
                                            }]
                                        },
                                        options: {
                                            scales: {
                                                x: { type: 'time', time: { unit: 'day' } },
                                                y: { beginAtZero: false }
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
        document.addEventListener('DOMContentLoaded', () => {
            const now = new Date().getHours();
            const rowH = parseInt(getComputedStyle(document.documentElement)
                .getPropertyValue('--row-height'));
            const container = document.getElementById('timetableContainer');
            container.scrollTop = rowH * now - container.clientHeight / 2;

            // fullscreen toggle
            const toggle = document.getElementById('toggleFullscreenBtn');
            const exitBtn = document.createElement('button');
            exitBtn.textContent = 'X';
            Object.assign(exitBtn.style, {
                position: 'absolute', top: '10px', right: '10px',
                background: '#dc3545', color: '#fff', border: 'none',
                padding: '6px 12px', cursor: 'pointer', display: 'none', zIndex: 10
            });
            container.appendChild(exitBtn);

            toggle.onclick = () => {
                Object.assign(container.style, {
                    position: 'fixed', top: 0, left: 0,
                    width: '100vw', height: '100vh'
                });
                toggle.style.display = 'none';
                exitBtn.style.display = 'block';
            };
            exitBtn.onclick = () => {
                Object.assign(container.style, {
                    position: '', top: '', left: '',
                    width: '', height: ''
                });
                exitBtn.style.display = 'none';
                toggle.style.display = '';
            };
        });
    </script>
</body>

</html>