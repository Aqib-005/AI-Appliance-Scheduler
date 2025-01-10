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
            $table->unsignedBigInteger('appliance_id');
            $table->string('name');
            $table->time('preferred_start');
            $table->time('preferred_end');
            $table->decimal('duration', 8, 2); // Add this line
            $table->json('usage_days');
            $table->time('predicted_start_time')->nullable();
            $table->time('predicted_end_time')->nullable();
            $table->timestamps();

            // Foreign key constraint
            $table->foreign('appliance_id')->references('id')->on('appliances')->onDelete('cascade');
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
