<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('appliances', function (Blueprint $table) {
            // Drop the usage_days column
            $table->dropColumn('usage_days');
        });

        Schema::table('appliances', function (Blueprint $table) {
            // Change preferred_start and preferred_end to time
            $table->time('preferred_start')->change();
            $table->time('preferred_end')->change();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('appliances', function (Blueprint $table) {
            // Re-add the usage_days column (if needed for rollback)
            $table->json('usage_days')->nullable()->default(null);
        });

        Schema::table('appliances', function (Blueprint $table) {
            // Revert to integer (if needed)
            $table->integer('preferred_start')->change();
            $table->integer('preferred_end')->change();
        });
    }
};
