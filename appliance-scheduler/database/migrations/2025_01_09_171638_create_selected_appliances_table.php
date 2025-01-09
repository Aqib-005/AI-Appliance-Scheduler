<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    /**
     * Run the migrations.
     */
    public function up()
    {
        Schema::create('selected_appliances', function (Blueprint $table) {
            $table->id();
            $table->string('name');
            $table->time('preferred_start');
            $table->time('preferred_end');
            $table->json('usage_days'); // Array of days
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('selected_appliances');
    }
};
