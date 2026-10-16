package org.mast3r.slam

import android.content.pm.ActivityInfo
import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Rule
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mast3r.slam.net.ScannerSession

@RunWith(AndroidJUnit4::class)
class ScanScreenTest {
    @get:Rule
    val compose = createComposeRule()

    @Before
    fun setContent() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        compose.setContent { MaterialTheme { ScanApp(ScannerSession(context)) } }
    }

    @Test
    fun landingPageShowsOnlyConnectionChoices() {
        compose.onNodeWithText("Connect to Linux").assertIsDisplayed()
        compose.onNodeWithText("Scan QR code").assertIsDisplayed()
        compose.onNodeWithText("Enter pairing URI").assertIsDisplayed()
        compose.onNodeWithText("Start scanning").assertDoesNotExist()
        compose.onNodeWithText("Pause").assertDoesNotExist()
        compose.onNodeWithText("View in AR").assertDoesNotExist()
    }

    @Test
    fun manualPairingRejectsInvalidUriClearly() {
        compose.onNodeWithText("Enter pairing URI").performClick()
        compose.onNodeWithText("Pairing URI").performTextInput("not-a-pairing-uri")
        compose.onNodeWithText("Connect").performClick()
        compose.onNodeWithText("Enter or scan a valid MASt3R pairing URI.").assertIsDisplayed()
    }

    @Test
    fun mainAndArActivitiesAreLandscapeOnly() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val main = context.packageManager.getActivityInfo(
            android.content.ComponentName(context, MainActivity::class.java),
            0,
        )
        val ar = context.packageManager.getActivityInfo(
            android.content.ComponentName(context, org.mast3r.slam.ar.ArActivity::class.java),
            0,
        )
        assertEquals(ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE, main.screenOrientation)
        assertEquals(ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE, ar.screenOrientation)
        assertEquals(ActivityInfo.LAUNCH_SINGLE_TASK, main.launchMode)
    }

    @Test
    fun mast3rOpenCvCoordinatesAreConvertedToArCoreAxes() {
        val converted = org.mast3r.slam.ar.MeshCoordinates.opencvToAr(
            floatArrayOf(3f, 4f, 5f),
            centerX = 1f,
            floorY = 10f,
            centerZ = 2f,
        )
        assertArrayEquals(floatArrayOf(2f, 6f, -3f), converted, 0.0001f)
    }

    @Test
    fun meshAlignmentIsHiddenUntilAlignedAndAfterTrackingLoss() {
        val bridge = org.mast3r.slam.net.ArAlignmentBridge
        bridge.reset()
        assertNull(bridge.modelMatrix)
        bridge.update("SEARCHING", FloatArray(0), "search")
        assertNull(bridge.modelMatrix)
        bridge.update("ALIGNED", FloatArray(16) { if (it % 5 == 0) 1f else 0f }, "aligned")
        assertNotNull(bridge.modelMatrix)
        bridge.trackingLost()
        assertNull(bridge.modelMatrix)
        bridge.trackingRestored()
        assertEquals("SEARCHING", bridge.status)
    }
}
