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
        Schema::create('appliances', function (Blueprint $table) {
            $table->id();
            $table->string('name');
            $table->decimal('power', 8, 2); // kW
            $table->integer('preferred_start'); // Hour (0-23)
            $table->integer('preferred_end'); // Hour (0-23)
            $table->decimal('duration', 8, 2); // Hours
            $table->json('usage_days'); // Days of the week
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('appliances');
    }
};
